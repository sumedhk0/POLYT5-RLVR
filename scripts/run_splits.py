"""Run the paper's FIVE-RANDOM-SPLIT protocol for Tg property prediction.

The paper reports property prediction as the mean over five random 80/20
train/test splits; the repository had only ever run split 0. This script runs
all five end to end and aggregates them into the mean +/- std format the paper
reports.

Per split k it: builds the k-th 80/20 split over the LamaLab Tg corpus,
fine-tunes with the config's paper hyperparameters (30 epochs, batch 16, AdamW,
lr 3e-4, weight decay 0.01), decodes the held-out 20% with BEAM SEARCH WIDTH 4,
and scores it with :func:`polyt5.evaluation.regression_report` (MAE primary,
plus RMSE, R^2, Pearson r; non-numeric decodes filtered and counted).

Each split's fine-tuned model is KEPT at ``split_<k>/checkpoints/best.pt``. The
five independently trained predictors are what
:class:`polyt5.inference.EnsemblePropertyPredictor` needs: inter-model
disagreement is an uncertainty estimate for the RL phase, and one member can be
reserved as a never-used-in-reward auditor.

Restartable: a split whose ``results.json`` and checkpoint already exist is
skipped unless ``--force``.

Usage:
    python scripts/run_splits.py \
        --config configs/finetune/tg_prediction.yaml \
        --init-checkpoint results/pretrain_small_pi1m/checkpoints/best.pt \
        --out results/tg_prediction_5splits

    # the paper's central ablation, over all five splits
    python scripts/run_splits.py --no-pretrained --out results/tg_prediction_5splits_scratch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from polyt5.data.collate import Seq2SeqCollator  # noqa: E402
from polyt5.data.datasets import Seq2SeqDataset  # noqa: E402
from polyt5.data.prepare import (  # noqa: E402
    build_prediction_examples,
    prepare_labeled_corpus,
    read_lamalab_tg,
)
from polyt5.data.splits import make_kfold_random_splits, save_splits  # noqa: E402
from polyt5.evaluation import (  # noqa: E402
    METRIC_NAMES,
    RegressionReport,
    aggregate_over_splits,
    regression_report,
)
from polyt5.generation import BeamSearchConfig, beam_search  # noqa: E402
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration  # noqa: E402
from polyt5.tokenization import PolyT5Tokenizer  # noqa: E402
from polyt5.training import Trainer, TrainerConfig, load_checkpoint  # noqa: E402
from polyt5.utils import (  # noqa: E402
    RunDirectory,
    describe_device,
    get_logger,
    load_config,
    parse_dotted_overrides,
    require,
    save_config,
    seed_everything,
    select_device,
)

RESULTS_FILENAME = "results.json"
AGGREGATE_FILENAME = "aggregate.json"


@dataclass(frozen=True)
class SplitIndices:
    """Index lists for one of the paper's random splits.

    Attributes:
        index: Which of the ``n_splits`` splits this is (0-based).
        train: Indices used for gradient updates.
        val: Indices carved out of the train pool for checkpoint selection.
            # [AMBIGUITY] The paper reports 80/20 train/test and names no
            # validation protocol (register F-01); an empty list here means "no
            # validation set", and then the last epoch's checkpoint is used.
        test: The held-out 20% the reported metrics are computed on.
    """

    index: int
    train: list[int]
    val: list[int]
    test: list[int]


def build_splits(
    n: int,
    *,
    n_splits: int = 5,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    base_seed: int = 0,
) -> list[SplitIndices]:
    """Construct the paper's five random splits, with validation carved out.

    The paper's protocol is five INDEPENDENT random 80/20 splits (seeds
    ``base_seed`` .. ``base_seed + n_splits - 1``), not a partitioning k-fold,
    so the five test sets legitimately overlap one another. Within one split,
    train, val and test are disjoint and together cover every index.

    Args:
        n: Dataset size.
        n_splits: How many splits to build. # [PAPER] 5.
        train_fraction: Train+val share of each split. # [PAPER] 0.8.
        val_fraction: Fraction OF THE TRAIN POOL held out for checkpoint
            selection. ``0.0`` disables validation entirely.
        base_seed: Seed of the first split.

    Returns:
        One :class:`SplitIndices` per split, in split order.

    Raises:
        ValueError: If ``n_splits`` is below 1 or ``val_fraction`` is outside
            ``[0, 1)``.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    folds = make_kfold_random_splits(
        n, k=n_splits, train_fraction=train_fraction, base_seed=base_seed
    )
    splits: list[SplitIndices] = []
    for index, (train_pool, test_idx) in enumerate(folds):
        n_val = round(len(train_pool) * val_fraction)
        val_idx, train_idx = train_pool[:n_val], train_pool[n_val:]
        splits.append(
            SplitIndices(index=index, train=train_idx, val=val_idx, test=test_idx)
        )
    return splits


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "finetune" / "tg_prediction.yaml")
    parser.add_argument("--splits", type=int, default=None,
                        help="Number of random splits (default: splits.n_splits, paper 5).")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="Pretrained polyT5 checkpoint used to initialise every split.")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Train every split from random initialisation (paper ablation).")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", type=Path, default=Path("results/tg_prediction_5splits"))
    parser.add_argument("--force", action="store_true",
                        help="Re-run splits whose results already exist.")
    parser.add_argument("--only-split", type=int, default=None,
                        help="Run just this split index (still aggregates what exists).")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Cap optimizer steps per split (debug).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of corpus rows read (debug).")
    return parser.parse_args(argv)


def _resolve(path_str: str | Path) -> Path:
    """Resolve a path relative to the repo root unless it is absolute."""
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def _build_model(
    cfg: dict[str, Any],
    tokenizer: PolyT5Tokenizer,
    args: argparse.Namespace,
    logger: Any,
    device: str,
) -> tuple[PolyT5ForConditionalGeneration, bool]:
    """Instantiate polyT5, optionally warm-started from a pretrained checkpoint.

    Mirrors ``scripts/finetune.py`` including its tokenizer-sha256 guard: a
    checkpoint trained on another vocabulary is refused, never silently remapped.

    Returns:
        ``(model_on_device, was_pretrained)``.
    """
    model_config_path = _resolve(
        cfg.get("model", {}).get("config") or cfg.get("model_config",
                                                      "configs/model/polyt5_small.yaml")
    )
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
        logger.info("split init: pretrained checkpoint %s (step %s)", init_ckpt,
                    state.get("global_step"))
        pretrained = True
    else:
        model_config.vocab_size = tokenizer.vocab_size
        model_config.pad_token_id = tokenizer.pad_id
        model_config.eos_token_id = tokenizer.eos_id
        model_config.decoder_start_token_id = tokenizer.decoder_start_token_id
        model = PolyT5ForConditionalGeneration(model_config)
        logger.info("split init: RANDOM (no pretraining) — the paper's ablation arm")
        pretrained = False

    return model.to(device), pretrained


@torch.no_grad()
def _decode_test_set(
    model: torch.nn.Module,
    tokenizer: PolyT5Tokenizer,
    pairs: list[tuple[str, str]],
    *,
    cfg: dict[str, Any],
    device: str,
    logger: Any,
) -> tuple[list[str], list[str]]:
    """Decode the held-out split with beam search.

    Args:
        model: The fine-tuned model.
        tokenizer: Its tokenizer.
        pairs: ``(source, target)`` test examples.
        cfg: Resolved run config (supplies batch size and beam width).
        device: Torch device string.
        logger: Progress logger.

    Returns:
        ``(targets, predictions)`` -- the ground-truth strings and the decoded
        model outputs, positionally aligned.
    """
    model.eval()
    sources = [source for source, _ in pairs]
    targets = [target for _, target in pairs]
    eval_cfg = cfg.get("evaluation", {})
    batch_size = int(eval_cfg.get("batch_size", 32))
    num_beams = int(eval_cfg.get("beam_width", eval_cfg.get("num_beams", 4)))  # [PAPER] 4
    max_source = int(cfg.get("data", {}).get("max_length", 200))

    predictions: list[str] = []
    for start in range(0, len(sources), batch_size):
        chunk = sources[start : start + batch_size]
        encoded = tokenizer.batch_encode(chunk, add_eos=True, max_length=max_source,
                                         padding=True, truncation=True)
        output = beam_search(
            model,
            torch.tensor(encoded["input_ids"], device=device),
            torch.tensor(encoded["attention_mask"], device=device),
            config=BeamSearchConfig(
                num_beams=num_beams,
                max_length=int(eval_cfg.get("max_target_length", 32)),
                length_penalty=float(eval_cfg.get("length_penalty", 1.0)),
                eos_token_id=tokenizer.eos_id,
                pad_token_id=tokenizer.pad_id,
                decoder_start_token_id=tokenizer.decoder_start_token_id,
            ),
        )
        predictions.extend(
            tokenizer.batch_decode(output.sequences.tolist(), skip_special_tokens=True)
        )
        logger.info("  decoded %d/%d", min(start + batch_size, len(sources)), len(sources))
    return targets, predictions


def _run_one_split(
    split: SplitIndices,
    examples: list[tuple[str, str]],
    *,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    tokenizer: PolyT5Tokenizer,
    tokenizer_path: Path,
    out_root: Path,
    device: str,
    logger: Any,
) -> dict[str, Any]:
    """Fine-tune, decode and score one split; persist everything it produced.

    Args:
        split: Index lists for this split.
        examples: All ``(PSELFIES, "236.0")`` examples, indexed by the split.
        cfg: Resolved run config.
        args: Parsed command line.
        tokenizer: Tokenizer shared by every split.
        tokenizer_path: Where that tokenizer lives, recorded in the checkpoint
            so :meth:`polyt5.inference.PolyT5PropertyPredictor.from_checkpoint`
            can find and verify it later.
        out_root: Root of the multi-split run directory.
        device: Torch device string.
        logger: Progress logger.

    Returns:
        The split's serialisable result payload.
    """
    run_dir = RunDirectory.create(out_root, f"split_{split.index}")
    train_pairs = [examples[i] for i in split.train]
    val_pairs = [examples[i] for i in split.val]
    test_pairs = [examples[i] for i in split.test]
    logger.info("split %d: %d train / %d val / %d test", split.index, len(train_pairs),
                len(val_pairs), len(test_pairs))

    train_cfg_raw = {**cfg.get("training", {}), **cfg.get("train", {})}
    max_source = int(train_cfg_raw.get("max_source_length", 200))
    max_target = int(train_cfg_raw.get("max_target_length", 200))
    collator = Seq2SeqCollator(pad_id=tokenizer.pad_id, max_source_length=max_source,
                               max_target_length=max_target)
    train_ds = Seq2SeqDataset(train_pairs, tokenizer, max_source_length=max_source,
                              max_target_length=max_target)
    val_ds = (
        Seq2SeqDataset(val_pairs, tokenizer, max_source_length=max_source,
                       max_target_length=max_target)
        if val_pairs
        else None
    )

    seed = int(cfg.get("seed", 0)) + split.index
    seed_everything(seed)
    model, pretrained = _build_model(cfg, tokenizer, args, logger, device)

    trainer_config = TrainerConfig(
        max_epochs=int(train_cfg_raw.get("epochs", train_cfg_raw.get("max_epochs", 30))),
        physical_batch_size=int(train_cfg_raw.get("batch_size",
                                                  train_cfg_raw.get("physical_batch_size", 16))),
        gradient_accumulation_steps=int(train_cfg_raw.get("gradient_accumulation_steps", 1)),
        learning_rate=float(train_cfg_raw.get("learning_rate", 3e-4)),   # [PAPER]
        weight_decay=float(train_cfg_raw.get("weight_decay", 0.01)),     # [PAPER]
        max_grad_norm=train_cfg_raw.get("max_grad_norm", 1.0),
        amp=bool(train_cfg_raw.get("amp", True)),
        amp_dtype=str(train_cfg_raw.get("amp_dtype", "bf16")),
        scheduler=str(train_cfg_raw.get("scheduler", "constant")),
        warmup_steps=int(train_cfg_raw.get("warmup_steps", 0)),
        log_every=int(train_cfg_raw.get("log_every", 50)),
        save_every_epochs=int(train_cfg_raw.get("save_every_epochs", 1)),
        keep_last_checkpoints=int(train_cfg_raw.get("keep_last_checkpoints", 1)),
        max_steps=args.max_steps,
        seed=seed,
        device=device,
        num_workers=int(train_cfg_raw.get("num_workers", 0)),
        target_effective_batch_size=train_cfg_raw.get("target_effective_batch_size"),
    )

    train_loader = DataLoader(train_ds, batch_size=trainer_config.physical_batch_size,
                              shuffle=True, collate_fn=collator,
                              num_workers=trainer_config.num_workers)
    val_loader = (
        DataLoader(val_ds, batch_size=trainer_config.physical_batch_size, shuffle=False,
                   collate_fn=collator, num_workers=trainer_config.num_workers)
        if val_ds is not None
        else None
    )

    save_config(cfg, run_dir.config_path)
    run_dir.write_manifest({
        "stage": "finetune_prediction_split",
        "split_index": split.index,
        "pretrained": pretrained,
        "tokenizer_sha256": tokenizer.sha256,
        "tokenizer_path": str(tokenizer_path),
        "n_train": len(train_pairs), "n_val": len(val_pairs), "n_test": len(test_pairs),
        "model_parameters": model.num_parameters(),
    })

    started = time.time()
    trainer = Trainer(model, train_loader, trainer_config, val_loader=val_loader,
                      run_dir=run_dir, tokenizer_path=tokenizer_path,
                      tokenizer_sha256=tokenizer.sha256, run_config=cfg, logger=logger)
    train_metrics = trainer.train()
    train_seconds = time.time() - started
    logger.info("split %d: trained in %.1fs — %s", split.index, train_seconds, train_metrics)

    # The Trainer only writes best.pt when a validation loss exists to select
    # on. With no validation split there is still a model worth keeping, so the
    # final-epoch weights are pinned under the same name; every split therefore
    # contributes exactly one ensemble member at a predictable path.
    checkpoint_path = run_dir.checkpoints / "best.pt"
    if not checkpoint_path.exists():
        trainer.save(path=checkpoint_path, train_metrics=train_metrics)
        logger.info("split %d: no validation set — pinned the final epoch as %s",
                    split.index, checkpoint_path)

    targets, predictions = _decode_test_set(model, tokenizer, test_pairs, cfg=cfg,
                                            device=device, logger=logger)
    report = regression_report(targets, predictions)
    run_dir.append_jsonl("predictions.jsonl", [
        {"source": source, "target": target, "prediction": prediction}
        for (source, target), prediction in zip(test_pairs, predictions, strict=True)
    ])

    # The checkpoint is deliberately KEPT: five splits give five independently
    # trained predictors, which is exactly what EnsemblePropertyPredictor needs.
    payload: dict[str, Any] = {
        "split_index": split.index,
        "seed": seed,
        "pretrained": pretrained,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "n_test": len(test_pairs),
        "train_seconds": train_seconds,
        "training": train_metrics,
        "evaluation": report.to_dict(),
        "checkpoint": str(checkpoint_path) if checkpoint_path.exists() else None,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": tokenizer.sha256,
    }
    run_dir.write_json(RESULTS_FILENAME, payload)
    logger.info("split %d: MAE=%s RMSE=%s R2=%s r=%s (non-numeric %.1f%%)", split.index,
                report.mae, report.rmse, report.r2, report.pearson_r,
                report.non_numeric_rate * 100)
    return payload


def _load_existing(out_root: Path, index: int) -> dict[str, Any] | None:
    """Return a previously written split payload, or ``None``.

    A split counts as complete only when both its ``results.json`` and its
    checkpoint survive, so a run interrupted between the two is redone rather
    than leaving the ensemble a member short.
    """
    results_path = out_root / f"split_{index}" / RESULTS_FILENAME
    if not results_path.is_file():
        return None
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    checkpoint = payload.get("checkpoint")
    if checkpoint and not Path(checkpoint).is_file():
        return None
    return payload


def _report_from_payload(payload: dict[str, Any]) -> RegressionReport | None:
    """Rebuild a :class:`RegressionReport` from a saved split payload."""
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        return None
    try:
        return RegressionReport(**evaluation)
    except TypeError:
        return None


def _format_aggregate(aggregate: dict[str, Any]) -> str:
    """Render mean +/- std per metric, the way the paper reports it."""
    lines = ["", f"Aggregate over {aggregate.get('n_splits', 0)} splits", "=" * 46]
    for name in METRIC_NAMES:
        entry = aggregate.get(name) or {}
        mean, std, count = entry.get("mean"), entry.get("std"), entry.get("n", 0)
        if mean is None:
            lines.append(f"{name:<16}{'n/a':>12}")
            continue
        spread = "n/a" if std is None else f"{std:.4f}"
        lines.append(f"{name:<16}{mean:>12.4f}  +/- {spread}   (n={count})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=parse_dotted_overrides(args.set))

    seed = int(cfg.get("seed", 0))
    seed_everything(seed)

    out_root = _resolve(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    logger = get_logger("polyt5.run_splits", log_file=out_root / "run_splits.log")

    device = select_device(cfg.get("train", {}).get("device", "auto"))
    logger.info("device=%s", describe_device(device).to_dict())

    tokenizer_path = _resolve(
        cfg.get("tokenizer", {}).get("path", "artifacts/tokenizer/polyt5_vocab.json")
    )
    tokenizer = PolyT5Tokenizer.from_file(tokenizer_path)

    csv_path = _resolve(require(cfg, "data.csv_path"))
    if not csv_path.is_file():
        logger.error("Tg corpus not found: %s — run scripts/download_data.py first", csv_path)
        return 1
    pairs, stats = prepare_labeled_corpus(
        read_lamalab_tg(csv_path, limit=args.limit),
        max_tokens=int(cfg.get("data", {}).get("max_length", 200)),
        deduplicate=bool(cfg.get("data", {}).get("deduplicate", True)),
        tokenizer=tokenizer,
    )
    if not pairs:
        logger.error("no rows survived preparation of %s; nothing to run", csv_path)
        return 1
    examples = build_prediction_examples(
        pairs, property_name=cfg.get("data", {}).get("property_name")
    )
    logger.info("corpus: %d usable rows (attrition %s)", len(examples),
                json.dumps(stats.to_dict()))

    n_splits = int(args.splits or require(cfg, "splits.n_splits"))
    splits = build_splits(
        len(examples),
        n_splits=n_splits,
        train_fraction=float(require(cfg, "splits.train_fraction")),
        val_fraction=float(cfg.get("splits", {}).get("val_fraction", 0.1)),
        base_seed=seed,
    )
    save_splits(out_root / "splits.json", {
        "task": "tg_prediction",
        "n": len(examples),
        "base_seed": seed,
        "n_splits": n_splits,
        "splits": [
            {"index": s.index, "train": s.train, "val": s.val, "test": s.test} for s in splits
        ],
    })

    payloads: list[dict[str, Any]] = []
    for split in splits:
        if args.only_split is not None and split.index != args.only_split:
            existing = _load_existing(out_root, split.index)
            if existing is not None:
                payloads.append(existing)
            continue

        existing = None if args.force else _load_existing(out_root, split.index)
        if existing is not None:
            logger.info("split %d/%d: already done — skipping (use --force to re-run)",
                        split.index + 1, len(splits))
            payloads.append(existing)
            continue

        logger.info("=" * 78)
        logger.info("split %d/%d starting", split.index + 1, len(splits))
        payloads.append(_run_one_split(
            split, examples, cfg=cfg, args=args, tokenizer=tokenizer,
            tokenizer_path=tokenizer_path, out_root=out_root, device=device, logger=logger,
        ))

    reports = [report for report in map(_report_from_payload, payloads) if report is not None]
    aggregate = aggregate_over_splits(reports)
    summary = {
        "config": str(args.config),
        "n_splits_requested": n_splits,
        "n_splits_completed": len(reports),
        "corpus_size": len(examples),
        "pretrained": not args.no_pretrained,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": tokenizer.sha256,
        # Kept so the RL phase can assemble an ensemble / reserve an auditor.
        "split_checkpoints": [payload.get("checkpoint") for payload in payloads],
        "per_split": [
            {"split_index": payload.get("split_index"), **(payload.get("evaluation") or {})}
            for payload in payloads
        ],
        "aggregate": aggregate,
    }
    (out_root / AGGREGATE_FILENAME).write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    logger.info("%s", _format_aggregate(aggregate))
    logger.info("wrote %s", out_root / AGGREGATE_FILENAME)
    return 0


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    raise SystemExit(main())
