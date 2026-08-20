"""Prepare the PSELFIES pretraining corpus from PI1M.

Usage:
    python scripts/prepare_pretraining_data.py --config configs/pretrain/dev.yaml

Reads the PI1M CSV, converts PSMILES -> PSELFIES with validity/length filters
and canonical deduplication, splits 90/5/5 (paper: 90/10 held out; the 5/5
sub-split is ours), and writes to ``data.output_dir``:

    train.txt / val.txt / test.txt   one PSELFIES per line
    stats.json                       attrition counters + provenance
    splits.json                      exact index lists, reproducible offline
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from polyt5.data.prepare import prepare_pselfies_corpus, read_pi1m  # noqa: E402
from polyt5.data.splits import make_pretraining_splits, save_splits  # noqa: E402
from polyt5.utils import load_config, require, seed_everything  # noqa: E402


def _resolve(path_str: str) -> Path:
    """Resolve a config path relative to the repo root unless absolute."""
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "pretrain" / "dev.yaml",
        help="pretraining YAML config (default: configs/pretrain/dev.yaml)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Tokenizer artifact JSON. When given, the 200-token length filter is applied to "
             "REAL tokenizer output rather than to a bracket-group count, which is what the "
             "model actually sees. Overrides any `tokenizer.path` in the config.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)

    tokenizer = None
    tokenizer_path = args.tokenizer or cfg.get("tokenizer", {}).get("path")
    if tokenizer_path is not None:
        from polyt5.tokenization import PolyT5Tokenizer

        tokenizer_path = _resolve(str(tokenizer_path))
        tokenizer = PolyT5Tokenizer.from_file(tokenizer_path)
        print(f"tokenizer: {tokenizer_path} (vocab={tokenizer.vocab_size}, "
              f"sha256={tokenizer.sha256[:16]})")
    else:
        print("tokenizer: none — length filter counts SELFIES bracket groups "
              "(see prepare._count_tokens)")

    csv_path = _resolve(require(cfg, "data.csv_path"))
    output_dir = _resolve(require(cfg, "data.output_dir"))
    limit = cfg["data"].get("limit")
    max_length = int(cfg["data"].get("max_length", 200))
    deduplicate = bool(cfg["data"].get("deduplicate", True))
    fractions = cfg.get("splits", {})

    print(f"config:  {args.config}")
    print(f"source:  {csv_path} (limit={limit})")
    pselfies, stats = prepare_pselfies_corpus(
        read_pi1m(csv_path, limit=limit),
        max_tokens=max_length,
        deduplicate=deduplicate,
        tokenizer=tokenizer,
        progress=sys.stderr.isatty(),
    )
    print("attrition:", json.dumps(stats.to_dict()))
    if stats.n_kept == 0:
        print("ERROR: no rows survived preparation; nothing written.")
        return 1

    splits = make_pretraining_splits(
        stats.n_kept,
        seed=seed,
        train=float(fractions.get("train", 0.9)),
        val=float(fractions.get("val", 0.05)),
        test=float(fractions.get("test", 0.05)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, indices in splits.items():
        out_path = output_dir / f"{split_name}.txt"
        with out_path.open("w", encoding="utf-8", newline="\n") as fh:
            for i in indices:
                fh.write(pselfies[i] + "\n")
        print(f"wrote {out_path}  ({len(indices)} lines)")

    save_splits(
        output_dir / "splits.json",
        {
            "seed": seed,
            "n": stats.n_kept,
            "fractions": {k: len(v) / stats.n_kept for k, v in splits.items()},
            **splits,
        },
    )
    stats_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "data_mode": cfg.get("data_mode", "proxy"),
        "source_csv": str(csv_path),
        "limit": limit,
        "max_length": max_length,
        "deduplicate": deduplicate,
        "seed": seed,
        # Which tokenizer defined the length filter. A corpus prepared with a
        # different vocabulary is a different corpus, so this is provenance,
        # not decoration.
        "tokenizer_path": str(tokenizer_path) if tokenizer is not None else None,
        "tokenizer_sha256": tokenizer.sha256 if tokenizer is not None else None,
        "length_filter_unit": (
            "tokenizer_ids" if tokenizer is not None else "selfies_bracket_groups"
        ),
        "attrition": stats.to_dict(),
        "split_sizes": {k: len(v) for k, v in splits.items()},
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {output_dir / 'splits.json'}")
    print(f"wrote {output_dir / 'stats.json'}")
    keep_rate = stats.n_kept / stats.n_input if stats.n_input else 0.0
    print(f"done: kept {stats.n_kept}/{stats.n_input} rows ({keep_rate:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
