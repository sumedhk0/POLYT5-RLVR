"""Span-corruption pretraining for polyT5.

This is the Stage 2+ entry point: it wires the prepared PSELFIES corpus, the
tokenizer artifact, the paper's span-corruption objective, and the Trainer into
one configured run. Everything experimental comes from the YAML config — nothing
is hard-coded here, including the paper's batch size of 450, which is expressed
as ``physical_batch_size x gradient_accumulation_steps`` and asserted against
``target_effective_batch_size``.

Two corpus backends
-------------------
``data.corpus_prefix``  A pre-tokenized, memory-mapped corpus built by
    ``scripts/prepare_large_corpus.py``. Takes precedence when present. Nothing
    is held in RAM, every DataLoader worker mmaps the same file, and the
    tokenizer never runs during training — which is what makes a 100M-polymer
    corpus trainable at all. The corpus records the SHA-256 of the vocabulary
    that produced its ids and this script REFUSES to train through a mismatch:
    the same id means a different token, so it would be a different corpus.

``data.processed_dir``  The original one-PSELFIES-per-line text corpus from
    ``scripts/prepare_pretraining_data.py``, tokenized per ``__getitem__``.
    Still supported, still the right thing for small runs.

Usage:
    python scripts/pretrain.py --config configs/pretrain/dev.yaml
    python scripts/pretrain.py --config configs/pretrain/polyone_medium.yaml
    python scripts/pretrain.py --config configs/pretrain/pi1m_small.yaml \
        --set train.physical_batch_size=8 train.gradient_accumulation_steps=56

Writes to ``results/<experiment_name>/``: config.yaml, manifest.json,
metrics.jsonl, metrics.csv, logs/, checkpoints/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]

from polyt5.data.collate import SpanCorruptionCollator  # noqa: E402
from polyt5.data.datasets import PSelfiesCorpus  # noqa: E402
from polyt5.data.span_corruption import SpanCorruptionConfig  # noqa: E402
from polyt5.data.tokenized_corpus import (  # noqa: E402
    MemmapPSelfiesDataset,
    TokenizedCorpus,
    TokenizerMismatchError,
    load_split_indices,
)
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration  # noqa: E402
from polyt5.tokenization import PolyT5Tokenizer  # noqa: E402
from polyt5.training import Trainer, TrainerConfig  # noqa: E402
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


def _read_corpus(path: Path, limit: int | None = None) -> list[str]:
    """Read one PSELFIES per line."""
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if text:
                lines.append(text)
            if limit is not None and len(lines) >= limit:
                break
    return lines


def _build_memmap_datasets(
    cfg: dict,
    tokenizer,
    *,
    max_length: int,
    limit_train: int | None,
    logger,
) -> tuple[MemmapPSelfiesDataset, MemmapPSelfiesDataset | None, dict[str, Any]]:
    """Open a pre-tokenized memory-mapped corpus and slice it into splits.

    Args:
        cfg: The full run config; reads ``data.corpus_prefix`` and, optionally,
            ``data.splits_path``.
        tokenizer: The configured tokenizer, checked against the corpus.
        max_length: Truncation applied on read (belt and braces; the builder
            already enforced the budget).
        limit_train: Debug cap on the number of training sequences.
        logger: Run logger.

    Returns:
        ``(train_dataset, val_dataset_or_None, info)`` where ``info`` holds the
        corpus statistics for the run manifest.

    Raises:
        TokenizerMismatchError: The corpus was built with another vocabulary.
        FileNotFoundError: The corpus or its splits file is missing.
    """
    data_cfg = cfg.get("data", {})
    prefix = _resolve(str(data_cfg["corpus_prefix"]))
    corpus = TokenizedCorpus.from_prefix(prefix)

    # A corpus tokenized with a different vocabulary is a different corpus:
    # the ids mean different tokens. Refuse rather than train on garbage.
    corpus.check_tokenizer(tokenizer)

    splits_path = data_cfg.get("splits_path")
    splits_path = _resolve(str(splits_path)) if splits_path else prefix.parent / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(
            f"no splits file at {splits_path}; run scripts/prepare_large_corpus.py "
            "or set data.splits_path"
        )
    train_idx = load_split_indices(splits_path, "train")
    try:
        val_idx: np.ndarray | None = load_split_indices(splits_path, "val")
    except KeyError:
        val_idx = None

    if limit_train is not None and train_idx.size > limit_train:
        train_idx = train_idx[:limit_train]

    lengths = corpus.lengths()
    logger.info(
        "corpus: %s — %d sequences, %d tokens, mean length %.1f (min %d, max %d), dtype=%s",
        prefix, len(corpus), corpus.n_tokens,
        float(lengths.mean()) if lengths.size else 0.0,
        int(lengths.min()) if lengths.size else 0,
        int(lengths.max()) if lengths.size else 0,
        corpus.dtype,
    )
    logger.info(
        "corpus attrition at build time: %s", corpus.metadata.get("preparation_stats", {})
    )
    logger.info(
        "splits: %s — %d train / %d val sequences",
        splits_path, train_idx.size, 0 if val_idx is None else val_idx.size,
    )

    train_ds = MemmapPSelfiesDataset(corpus, train_idx, max_length=max_length)
    val_ds = (
        MemmapPSelfiesDataset(corpus, val_idx, max_length=max_length)
        if val_idx is not None and val_idx.size
        else None
    )
    info = {
        "corpus_mode": "memmap",
        "corpus_prefix": str(prefix),
        "corpus_dir": str(prefix.parent),
        "splits_path": str(splits_path),
        "n_corpus_sequences": len(corpus),
        "n_corpus_tokens": corpus.n_tokens,
        "corpus_mean_length": float(lengths.mean()) if lengths.size else 0.0,
        "corpus_preparation_stats": corpus.metadata.get("preparation_stats", {}),
        "corpus_source_path": corpus.metadata.get("source_path"),
    }
    return train_ds, val_ds, info


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "pretrain" / "dev.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Dotted config overrides, e.g. train.physical_batch_size=8")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Checkpoint to resume from.")
    parser.add_argument("--limit-train", type=int, default=None,
                        help="Debug: cap the number of training sequences.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Debug: stop after this many optimizer steps.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=parse_dotted_overrides(args.set))

    seed = int(cfg.get("seed", 0))
    seed_everything(seed)

    experiment = cfg.get("experiment_name") or cfg.get("name") or args.config.stem
    results_root = _resolve(cfg.get("output", {}).get("results_root", "results"))
    run_dir = RunDirectory.create(results_root, experiment)
    logger = get_logger("polyt5.pretrain", log_file=run_dir.logs / "pretrain.log")

    device = select_device(cfg.get("train", {}).get("device", "auto"))
    device_info = describe_device(device)
    logger.info("device: %s", device_info.to_dict())

    # ------------------------------------------------------------ tokenizer
    tokenizer_path = _resolve(
        cfg.get("tokenizer", {}).get("path", "artifacts/tokenizer/polyt5_vocab.json")
    )
    tokenizer = PolyT5Tokenizer.from_file(tokenizer_path)
    logger.info("tokenizer: %s vocab=%d sha256=%s", tokenizer_path, tokenizer.vocab_size,
                tokenizer.sha256[:16])

    # ----------------------------------------------------------------- data
    data_cfg = cfg.get("data", {})
    max_length = int(data_cfg.get("max_length", 200))

    if data_cfg.get("corpus_prefix"):
        # Pre-tokenized memory-mapped corpus (scripts/prepare_large_corpus.py).
        try:
            train_ds, val_ds, corpus_info = _build_memmap_datasets(
                cfg, tokenizer, max_length=max_length,
                limit_train=args.limit_train, logger=logger,
            )
        except TokenizerMismatchError as exc:
            logger.error("refusing to train: %s", exc)
            return 1
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 1
        if len(train_ds) == 0:
            logger.error("the memmap corpus has no training sequences")
            return 1
    else:
        # Original plain-text corpus (scripts/prepare_pretraining_data.py).
        processed_dir = _resolve(data_cfg.get("processed_dir") or cfg["data"]["output_dir"])
        train_texts = _read_corpus(processed_dir / "train.txt", limit=args.limit_train)
        val_path = processed_dir / "val.txt"
        val_texts = _read_corpus(val_path) if val_path.exists() else []
        if not train_texts:
            logger.error("no training sequences found in %s — run "
                         "scripts/prepare_pretraining_data.py first", processed_dir)
            return 1
        logger.info("corpus: %d train / %d val sequences from %s", len(train_texts),
                    len(val_texts), processed_dir)
        train_ds = PSelfiesCorpus(train_texts, tokenizer, max_length=max_length)
        val_ds = PSelfiesCorpus(val_texts, tokenizer, max_length=max_length) if val_texts else None
        corpus_info = {
            "corpus_mode": "text",
            "corpus_dir": str(processed_dir),
        }

    span_cfg_raw = cfg.get("span_corruption", {})
    span_config = SpanCorruptionConfig(
        max_spans=int(span_cfg_raw.get("max_spans", 8)),
        max_span_length=int(span_cfg_raw.get("max_span_length", 3)),
        corruption_rate=float(span_cfg_raw.get("corruption_rate", 0.15)),
        min_gap=int(span_cfg_raw.get("min_gap", 1)),
        add_final_sentinel=bool(span_cfg_raw.get("add_final_sentinel", True)),
        mask_eos=bool(span_cfg_raw.get("mask_eos", False)),
    )
    logger.info("span corruption: %s", span_config)

    collator = SpanCorruptionCollator(
        sentinel_ids=tokenizer.sentinel_ids,
        pad_id=tokenizer.pad_id,
        eos_id=tokenizer.eos_id,
        config=span_config,
        max_length=max_length,
        seed=seed,
    )

    # Configs in this repo use either `train:` or `training:` for the same
    # block. Merge both rather than letting one silently shadow the other --
    # a `--set train.foo=1` override on a `training:`-style config would
    # otherwise create an almost-empty `train:` block and hide the real one.
    train_cfg_raw = {**cfg.get("training", {}), **cfg.get("train", {})}
    trainer_config = TrainerConfig(
        max_epochs=int(train_cfg_raw.get("max_epochs", 1)),
        physical_batch_size=int(train_cfg_raw["physical_batch_size"]),
        gradient_accumulation_steps=int(train_cfg_raw["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg_raw.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg_raw.get("weight_decay", 0.0)),
        max_grad_norm=train_cfg_raw.get("max_grad_norm", 1.0),
        amp=bool(train_cfg_raw.get("amp", True)),
        amp_dtype=str(train_cfg_raw.get("amp_dtype", "bf16")),
        scheduler=str(train_cfg_raw.get("scheduler", "inverse_sqrt")),
        warmup_steps=int(train_cfg_raw.get("warmup_steps", 0)),
        log_every=int(train_cfg_raw.get("log_every", 50)),
        save_every_epochs=int(train_cfg_raw.get("save_every_epochs", 1)),
        keep_last_checkpoints=int(train_cfg_raw.get("keep_last_checkpoints", 3)),
        max_steps=args.max_steps if args.max_steps is not None
        else train_cfg_raw.get("max_steps"),
        seed=seed,
        device=device,
        num_workers=int(train_cfg_raw.get("num_workers", 0)),
        target_effective_batch_size=train_cfg_raw.get("target_effective_batch_size"),
    )
    logger.info("effective batch size: %d = %d x %d", trainer_config.effective_batch_size,
                trainer_config.physical_batch_size, trainer_config.gradient_accumulation_steps)

    train_loader = DataLoader(
        train_ds,
        batch_size=trainer_config.physical_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=trainer_config.num_workers,
        drop_last=False,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=trainer_config.physical_batch_size, shuffle=False,
                   collate_fn=collator, num_workers=trainer_config.num_workers)
        if val_ds is not None else None
    )

    # ---------------------------------------------------------------- model
    model_config = PolyT5Config.from_yaml(_resolve(
        cfg.get("model", {}).get("config") or cfg["model_config"]
    ))
    model_config.vocab_size = tokenizer.vocab_size
    model_config.pad_token_id = tokenizer.pad_id
    model_config.eos_token_id = tokenizer.eos_id
    model_config.decoder_start_token_id = tokenizer.decoder_start_token_id
    model = PolyT5ForConditionalGeneration(model_config).to(device)
    logger.info("model: %d parameters (d_model=%d, %d+%d layers)", model.num_parameters(),
                model_config.d_model, model_config.num_layers,
                model_config.num_decoder_layers_resolved)

    # -------------------------------------------------------------- training
    save_config(cfg, run_dir.config_path)
    run_dir.write_manifest({
        "stage": "pretrain",
        "experiment": experiment,
        "device": device_info.to_dict(),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": tokenizer.sha256,
        "n_train_sequences": len(train_ds),
        "n_val_sequences": 0 if val_ds is None else len(val_ds),
        "model_parameters": model.num_parameters(),
        "effective_batch_size": trainer_config.effective_batch_size,
        "data_mode": cfg.get("data_mode", "proxy"),
        **corpus_info,
    })

    trainer = Trainer(
        model,
        train_loader,
        trainer_config,
        val_loader=val_loader,
        run_dir=run_dir,
        tokenizer_path=str(tokenizer_path),
        tokenizer_sha256=tokenizer.sha256,
        run_config=cfg,
        logger=logger,
    )
    if args.resume is not None:
        trainer.resume_from(_resolve(args.resume))
        logger.info("resumed from %s", args.resume)

    metrics = trainer.train()
    summary = trainer.summary()
    run_dir.write_json("summary.json", {"metrics": metrics, "summary": summary})

    logger.info("=" * 78)
    for key, value in summary.items():
        logger.info("%-28s %s", key, value)
    logger.info("final metrics: %s", metrics)
    logger.info("results in %s", run_dir.root)
    return 0


if __name__ == "__main__":
    if torch.cuda.is_available():
        # TF32 matmuls on Ada: a large speedup at negligible cost for this
        # workload. The paper states no precision policy (register E-04).
        torch.set_float32_matmul_precision("high")
    sys.exit(main())
