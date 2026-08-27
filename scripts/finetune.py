"""Supervised fine-tuning of polyT5 on the two downstream tasks.

Both tasks are sequence-to-sequence and share this entry point; they differ in
their I/O direction and in how they are decoded at evaluation time.

    --task prediction   PSELFIES -> "236.0"      decoded with BEAM SEARCH (width 4)
    --task generation   "500.0"  -> PSELFIES     decoded by SAMPLING (top-p, temperature)

Both formats are reproduced verbatim from the paper's Supplementary Information.
The paper uses beam search for property prediction and states explicitly that
generation used "a sampling-based approach instead of beam search".

Paper hyperparameters (both tasks): AdamW, lr 3e-4, weight decay 0.01, batch
size 16, token-level cross-entropy with padded labels set to -100, evaluation at
the end of each epoch. Prediction runs up to 30 epochs on an 80/20 split over
five random splits; generation runs up to 15 epochs on a 90/10 split.

Usage:
    python scripts/finetune.py --task prediction \
        --config configs/finetune/tg_prediction.yaml \
        --init-checkpoint results/pretrain_small_dev/checkpoints/best.pt

    # the paper's central ablation: identical run, randomly initialised weights
    python scripts/finetune.py --task prediction \
        --config configs/finetune/tg_prediction.yaml --no-pretrained
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]

from polyt5.data.collate import Seq2SeqCollator  # noqa: E402
from polyt5.data.datasets import Seq2SeqDataset  # noqa: E402
from polyt5.evaluation import (  # noqa: E402
    evaluate_generation,
    format_console_summary,
    regression_report,
)
from polyt5.generation import (  # noqa: E402
    BeamSearchConfig,
    GenerationConfig,
    beam_search,
    generate,
)
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration  # noqa: E402
from polyt5.tokenization import PolyT5Tokenizer  # noqa: E402
from polyt5.training import Trainer, TrainerConfig, load_checkpoint  # noqa: E402
from polyt5.utils import (  # noqa: E402
    # noqa: E402,
    RunDirectory,
    describe_device,
    get_logger,
    load_config,
    parse_dotted_overrides,
    resolve_under,
    save_config,
    seed_everything,
    select_device,
)


def _resolve(path_str: str | Path) -> Path:
    return resolve_under(REPO_ROOT, path_str)


def _read_jsonl(path: Path) -> list[tuple[str, str]]:
    """Read ``{"source": ..., "target": ...}`` lines into (source, target) pairs."""
    pairs: list[tuple[str, str]] = []
    if not path.exists():
        return pairs
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pairs.append((row["source"], row["target"]))
    return pairs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=["prediction", "generation"], required=True)
    parser.add_argument("--config", type=Path, default=None,
                        help="Defaults to configs/finetune/tg_<task>.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="Pretrained polyT5 checkpoint used to initialise the policy.")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Train from random initialisation. This is the paper's central "
                             "ablation — it reports Tg RMSE degrading 40.82 -> 89.35 without "
                             "pretraining.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="Cap the number of evaluation examples (debug).")
    return parser.parse_args(argv)


def _build_model(cfg, tokenizer, args, logger, device):
    """Instantiate polyT5, optionally warm-started from a pretrained checkpoint."""
    model_config_path = _resolve(cfg.get("model", {}).get("config")
                                 or cfg.get("model_config", "configs/model/polyt5_small.yaml"))
    model_config = PolyT5Config.from_yaml(model_config_path)
    init_ckpt = None if args.no_pretrained else (
        args.init_checkpoint or cfg.get("model", {}).get("init_checkpoint")
    )

    if init_ckpt is not None:
        state = load_checkpoint(_resolve(init_ckpt), map_location="cpu")
        ckpt_sha = state.get("tokenizer_sha256")
        if ckpt_sha and ckpt_sha != tokenizer.sha256:
            raise ValueError(
                "tokenizer mismatch: the checkpoint was trained with vocabulary "
                f"{ckpt_sha[:16]} but the configured tokenizer is {tokenizer.sha256[:16]}. "
                "Fine-tuning across vocabularies would silently corrupt every token id."
            )
        model_config = PolyT5Config.from_dict(state["model_config"])
        model = PolyT5ForConditionalGeneration(model_config)
        model.load_state_dict(state["model_state"])
        logger.info("initialised from pretrained checkpoint %s (step %s)", init_ckpt,
                    state.get("global_step"))
        pretrained = True
    else:
        model_config.vocab_size = tokenizer.vocab_size
        model_config.pad_token_id = tokenizer.pad_id
        model_config.eos_token_id = tokenizer.eos_id
        model_config.decoder_start_token_id = tokenizer.decoder_start_token_id
        model = PolyT5ForConditionalGeneration(model_config)
        logger.info("initialised RANDOMLY (no pretraining) — this is the paper's ablation arm")
        pretrained = False

    return model.to(device), model_config, pretrained


@torch.no_grad()
def _decode_test_set(model, tokenizer, pairs, *, task, cfg, device, logger, limit=None):
    """Decode the held-out set the way the paper decodes it for this task."""
    model.eval()
    pairs = pairs[:limit] if limit else pairs
    sources = [s for s, _ in pairs]
    targets = [t for _, t in pairs]
    eval_cfg = cfg.get("evaluation", {})
    batch_size = int(eval_cfg.get("batch_size", 32))
    max_length = int(cfg.get("data", {}).get("max_length", 200))

    predictions: list[str] = []
    for start in range(0, len(sources), batch_size):
        chunk = sources[start:start + batch_size]
        encoded = tokenizer.batch_encode(chunk, add_eos=True, max_length=max_length,
                                         padding=True, truncation=True)
        input_ids = torch.tensor(encoded["input_ids"], device=device)
        attention_mask = torch.tensor(encoded["attention_mask"], device=device)

        if task == "prediction":
            # [PAPER] beam search, width 4
            output = beam_search(
                model, input_ids, attention_mask,
                config=BeamSearchConfig(
                    num_beams=int(eval_cfg.get("num_beams", 4)),
                    max_length=int(eval_cfg.get("max_target_length", 32)),
                    length_penalty=float(eval_cfg.get("length_penalty", 1.0)),
                    eos_token_id=tokenizer.eos_id,
                    pad_token_id=tokenizer.pad_id,
                    decoder_start_token_id=tokenizer.decoder_start_token_id,
                ),
            )
        else:
            # [PAPER] sampling, explicitly "instead of beam search"
            output = generate(
                model, input_ids, attention_mask,
                config=GenerationConfig(
                    max_length=max_length,
                    do_sample=True,
                    temperature=float(eval_cfg.get("temperature", 1.1)),
                    top_p=float(eval_cfg.get("top_p", 0.75)),
                    eos_token_id=tokenizer.eos_id,
                    pad_token_id=tokenizer.pad_id,
                    decoder_start_token_id=tokenizer.decoder_start_token_id,
                    seed=int(cfg.get("seed", 0)) + start,
                ),
            )
        predictions.extend(
            tokenizer.batch_decode(output.sequences.tolist(), skip_special_tokens=True)
        )
        logger.info("decoded %d/%d", min(start + batch_size, len(sources)), len(sources))

    return sources, targets, predictions


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config or (REPO_ROOT / "configs" / "finetune" / f"tg_{args.task}.yaml")
    cfg = load_config(config_path, overrides=parse_dotted_overrides(args.set))

    seed = int(cfg.get("seed", 0))
    seed_everything(seed)

    suffix = "_scratch" if args.no_pretrained else ""
    experiment = (cfg.get("experiment_name") or cfg.get("name")
                  or f"finetune_tg_{args.task}") + suffix
    run_dir = RunDirectory.create(_resolve(cfg.get("output", {}).get("results_root", "results")),
                                  experiment)
    logger = get_logger("polyt5.finetune", log_file=run_dir.logs / "finetune.log")

    device = select_device(cfg.get("train", {}).get("device", "auto"))
    logger.info("task=%s device=%s", args.task, describe_device(device).to_dict())

    tokenizer = PolyT5Tokenizer.from_file(_resolve(
        cfg.get("tokenizer", {}).get("path", "artifacts/tokenizer/polyt5_vocab.json")
    ))

    data_dir = _resolve(cfg.get("data", {}).get("processed_dir")
                        or f"data/processed/tg/{args.task}")
    train_pairs = _read_jsonl(data_dir / "train.jsonl")
    val_pairs = _read_jsonl(data_dir / "val.jsonl")
    test_pairs = _read_jsonl(data_dir / "test.jsonl") or val_pairs
    if not train_pairs:
        logger.error("no training data in %s — run scripts/prepare_tg_data.py first", data_dir)
        return 1
    logger.info("data: %d train / %d val / %d test from %s", len(train_pairs), len(val_pairs),
                len(test_pairs), data_dir)
    logger.info("example: source=%r target=%r", *train_pairs[0])

    max_source = int(cfg.get("data", {}).get("max_source_length", 200))
    max_target = int(cfg.get("data", {}).get("max_target_length", 200))
    collator = Seq2SeqCollator(pad_id=tokenizer.pad_id, max_source_length=max_source,
                               max_target_length=max_target)
    train_ds = Seq2SeqDataset(train_pairs, tokenizer, max_source_length=max_source,
                              max_target_length=max_target)
    val_ds = Seq2SeqDataset(val_pairs, tokenizer, max_source_length=max_source,
                            max_target_length=max_target) if val_pairs else None

    model, model_config, pretrained = _build_model(cfg, tokenizer, args, logger, device)
    logger.info("model: %d parameters", model.num_parameters())

    train_cfg_raw = {**cfg.get("training", {}), **cfg.get("train", {})}
    trainer_config = TrainerConfig(
        max_epochs=int(train_cfg_raw.get("max_epochs", 30 if args.task == "prediction" else 15)),
        physical_batch_size=int(train_cfg_raw.get("physical_batch_size", 16)),  # [PAPER] 16
        gradient_accumulation_steps=int(train_cfg_raw.get("gradient_accumulation_steps", 1)),
        learning_rate=float(train_cfg_raw.get("learning_rate", 3e-4)),          # [PAPER]
        weight_decay=float(train_cfg_raw.get("weight_decay", 0.01)),            # [PAPER]
        max_grad_norm=train_cfg_raw.get("max_grad_norm", 1.0),
        amp=bool(train_cfg_raw.get("amp", True)),
        amp_dtype=str(train_cfg_raw.get("amp_dtype", "bf16")),
        scheduler=str(train_cfg_raw.get("scheduler", "constant")),
        warmup_steps=int(train_cfg_raw.get("warmup_steps", 0)),
        log_every=int(train_cfg_raw.get("log_every", 50)),
        save_every_epochs=int(train_cfg_raw.get("save_every_epochs", 1)),
        keep_last_checkpoints=int(train_cfg_raw.get("keep_last_checkpoints", 3)),
        max_steps=args.max_steps,
        seed=seed,
        device=device,
        num_workers=int(train_cfg_raw.get("num_workers", 0)),
        target_effective_batch_size=train_cfg_raw.get("target_effective_batch_size"),
    )

    train_loader = DataLoader(train_ds, batch_size=trainer_config.physical_batch_size,
                              shuffle=True, collate_fn=collator,
                              num_workers=trainer_config.num_workers)
    val_loader = (DataLoader(val_ds, batch_size=trainer_config.physical_batch_size, shuffle=False,
                             collate_fn=collator, num_workers=trainer_config.num_workers)
                  if val_ds is not None else None)

    save_config(cfg, run_dir.config_path)
    run_dir.write_manifest({
        "stage": f"finetune_{args.task}",
        "pretrained": pretrained,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "tokenizer_sha256": tokenizer.sha256,
        "data_dir": str(data_dir),
        "n_train": len(train_pairs), "n_val": len(val_pairs), "n_test": len(test_pairs),
        "model_parameters": model.num_parameters(),
        "data_mode": cfg.get("data_mode", "proxy"),
    })

    trainer = Trainer(model, train_loader, trainer_config, val_loader=val_loader,
                      run_dir=run_dir, tokenizer_sha256=tokenizer.sha256, run_config=cfg,
                      logger=logger)
    metrics = trainer.train()
    logger.info("training done: %s", metrics)

    # ------------------------------------------------------------- evaluation
    sources, targets, predictions = _decode_test_set(
        model, tokenizer, test_pairs, task=args.task, cfg=cfg, device=device, logger=logger,
        limit=args.eval_limit,
    )

    if args.task == "prediction":
        report = regression_report(targets, predictions)
        payload = report.to_dict()
        run_dir.append_jsonl("predictions.jsonl", [
            {"source": s, "target": t, "prediction": p}
            for s, t, p in zip(sources, targets, predictions, strict=True)
        ])
    else:
        report = evaluate_generation(predictions, target_property=None)
        payload = report.to_dict()
        run_dir.append_jsonl("generations.jsonl", [
            {"target_property": s, "generated": p}
            for s, p in zip(sources, predictions, strict=True)
        ])

    payload["pretrained"] = pretrained
    payload["n_evaluated"] = len(predictions)
    run_dir.write_json("metrics.json", {"training": metrics, "evaluation": payload})

    logger.info("=" * 78)
    try:
        logger.info("\n%s", format_console_summary(report))
    except Exception:  # regression reports may not be supported by the formatter
        logger.info("evaluation: %s", json.dumps(payload, indent=2, default=str))
    logger.info("results in %s", run_dir.root)
    return 0


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    sys.exit(main())
