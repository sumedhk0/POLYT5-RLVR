"""Prepare BOTH Tg task datasets (prediction and generation) from the LamaLab CSV.

Usage:
    python scripts/prepare_tg_data.py --config configs/finetune/tg_prediction.yaml

Reads the LamaLab curated Tg CSV once, converts PSMILES -> PSELFIES with the
shared filter pipeline, then writes both task formats (paper SI, verbatim):

    data/processed/tg/prediction/{train,val,test}.jsonl
        {"source": "<PSELFIES>", "target": "236.0"}
        80/20 train/test over five random splits (paper); the materialized
        files use ``splits.split_index``, with validation carved from the
        training portion (ours -- register F-01).
    data/processed/tg/generation/{train,val,test}.jsonl
        {"source": "236.0", "target": "<PSELFIES>"}
        90/10 train/validation (paper). The paper defines no generation test
        set, so test.jsonl is written empty and stats.json says why.

Each task directory also gets stats.json (provenance + attrition) and
splits.json (exact index lists).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from polyt5.data.prepare import (  # noqa: E402
    build_generation_examples,
    build_prediction_examples,
    prepare_labeled_corpus,
    read_lamalab_tg,
)
from polyt5.data.splits import make_kfold_random_splits, random_split, save_splits  # noqa: E402
from polyt5.utils import load_config, require, seed_everything  # noqa: E402


def _resolve(path_str: str) -> Path:
    """Resolve a config path relative to the repo root unless absolute."""
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def _write_jsonl(path: Path, examples: list[tuple[str, str]]) -> None:
    """Write (source, target) pairs as JSON lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for source, target in examples:
            fh.write(json.dumps({"source": source, "target": target}) + "\n")
    print(f"wrote {path}  ({len(examples)} examples)")


def _write_stats(path: Path, payload: dict) -> None:
    """Write a stats/provenance sidecar."""
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "finetune" / "tg_prediction.yaml",
        help="prediction YAML config (default: configs/finetune/tg_prediction.yaml)",
    )
    parser.add_argument(
        "--generation-config",
        type=Path,
        default=REPO_ROOT / "configs" / "finetune" / "tg_generation.yaml",
        help="generation YAML config supplying the 90/10 split parameters",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Tokenizer artifact JSON. When given, the 200-token length filter is applied to REAL "
             "tokenizer output rather than to a bracket-group count. Overrides `tokenizer.path`.",
    )
    args = parser.parse_args(argv)

    pred_cfg = load_config(args.config)
    gen_cfg = load_config(args.generation_config)
    seed = int(pred_cfg.get("seed", 0))
    seed_everything(seed)

    tokenizer = None
    tokenizer_path = args.tokenizer or pred_cfg.get("tokenizer", {}).get("path")
    if tokenizer_path is not None:
        from polyt5.tokenization import PolyT5Tokenizer

        tokenizer_path = _resolve(str(tokenizer_path))
        tokenizer = PolyT5Tokenizer.from_file(tokenizer_path)
        print(f"tokenizer: {tokenizer_path} (vocab={tokenizer.vocab_size}, "
              f"sha256={tokenizer.sha256[:16]})")

    csv_path = _resolve(require(pred_cfg, "data.csv_path"))
    output_root = _resolve(require(pred_cfg, "data.output_dir"))
    max_length = int(pred_cfg["data"].get("max_length", 200))
    deduplicate = bool(pred_cfg["data"].get("deduplicate", True))

    print(f"config:  {args.config}")
    print(f"source:  {csv_path}")
    pairs, stats = prepare_labeled_corpus(
        read_lamalab_tg(csv_path),
        max_tokens=max_length,
        deduplicate=deduplicate,
        tokenizer=tokenizer,
        progress=sys.stderr.isatty(),
    )
    print("attrition:", json.dumps(stats.to_dict()))
    if stats.n_kept == 0:
        print("ERROR: no rows survived preparation; nothing written.")
        return 1
    n = len(pairs)

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_mode": pred_cfg.get("data_mode", "proxy"),
        "source_csv": str(csv_path),
        "property": pred_cfg["data"].get("property_name", "Tg"),
        "max_length": max_length,
        "deduplicate": deduplicate,
        "seed": seed,
        "attrition": stats.to_dict(),
    }

    # ------------------------------------------------------------ prediction
    n_splits = int(require(pred_cfg, "splits.n_splits"))
    train_fraction = float(require(pred_cfg, "splits.train_fraction"))
    split_index = int(pred_cfg["splits"].get("split_index", 0))
    val_fraction = float(pred_cfg["splits"].get("val_fraction", 0.1))

    folds = make_kfold_random_splits(
        n, k=n_splits, train_fraction=train_fraction, base_seed=seed
    )
    train_pool, test_idx = folds[split_index]
    # [AMBIGUITY] The paper reports 80/20 train/test only; we hold out
    # val_fraction of the train pool for checkpoint selection (register F-01).
    n_val = round(len(train_pool) * val_fraction)
    val_idx, train_idx = train_pool[:n_val], train_pool[n_val:]

    pred_dir = output_root / "prediction"
    pred_examples = build_prediction_examples(
        pairs, property_name=pred_cfg["data"].get("property_name")
    )
    _write_jsonl(pred_dir / "train.jsonl", [pred_examples[i] for i in train_idx])
    _write_jsonl(pred_dir / "val.jsonl", [pred_examples[i] for i in val_idx])
    _write_jsonl(pred_dir / "test.jsonl", [pred_examples[i] for i in test_idx])
    save_splits(
        pred_dir / "splits.json",
        {
            "task": "tg_prediction",
            "n": n,
            "base_seed": seed,
            "n_splits": n_splits,
            "train_fraction": train_fraction,
            "split_index": split_index,
            "val_fraction": val_fraction,
            "materialized": {
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
            },
            "all_folds": [{"train": tr, "test": te} for tr, te in folds],
        },
    )
    _write_stats(
        pred_dir / "stats.json",
        {
            **provenance,
            "task": "tg_prediction",
            "config": str(args.config),
            "format": {"source": "bare PSELFIES (no task prefix)", "target": "'236.0'"},
            "split_sizes": {
                "train": len(train_idx),
                "val": len(val_idx),
                "test": len(test_idx),
            },
        },
    )

    # ------------------------------------------------------------ generation
    gen_train_frac = float(gen_cfg.get("splits", {}).get("train", 0.9))
    gen_val_frac = float(gen_cfg.get("splits", {}).get("val", 0.1))
    gen_train_idx, gen_val_idx = random_split(n, [gen_train_frac, gen_val_frac], seed=seed)

    gen_dir = output_root / "generation"
    gen_examples = build_generation_examples(pairs)
    _write_jsonl(gen_dir / "train.jsonl", [gen_examples[i] for i in gen_train_idx])
    _write_jsonl(gen_dir / "val.jsonl", [gen_examples[i] for i in gen_val_idx])
    # [AMBIGUITY] The paper's generation task has train/validation only; its
    # final evaluation samples the model and scores with the predictor. An
    # empty test.jsonl keeps the file contract without leaking val data.
    _write_jsonl(gen_dir / "test.jsonl", [])
    save_splits(
        gen_dir / "splits.json",
        {
            "task": "tg_generation",
            "n": n,
            "seed": seed,
            "train_fraction": gen_train_frac,
            "val_fraction": gen_val_frac,
            "train": gen_train_idx,
            "val": gen_val_idx,
            "test": [],
        },
    )
    _write_stats(
        gen_dir / "stats.json",
        {
            **provenance,
            "task": "tg_generation",
            "config": str(args.generation_config),
            "format": {"source": "'236.0'", "target": "bare PSELFIES"},
            "split_sizes": {
                "train": len(gen_train_idx),
                "val": len(gen_val_idx),
                "test": 0,
            },
            "note": (
                "test.jsonl is intentionally empty: the paper defines no test split "
                "for conditional generation (90/10 train/validation only)."
            ),
        },
    )

    keep_rate = stats.n_kept / stats.n_input if stats.n_input else 0.0
    print(f"done: kept {stats.n_kept}/{stats.n_input} rows ({keep_rate:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
