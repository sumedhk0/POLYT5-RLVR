#!/usr/bin/env python
"""Build the polyT5 tokenizer vocabulary artifact.

Writes a deterministic JSON artifact (``{version, sha256, metadata, tokens}``) that every
downstream stage -- pre-training, fine-tuning, evaluation, generation and the RLVR reward
layer -- loads via :meth:`PolyT5Tokenizer.from_file`. Re-running with the same arguments
produces a byte-identical file, so the sha256 printed here is a stable build fingerprint.

Examples:
    Default paper-shaped 458-token vocabulary::

        python scripts/build_tokenizer.py

    Base block ranked by frequency over a real PSELFIES corpus, plus a SentencePiece side
    file for interop::

        python scripts/build_tokenizer.py --corpus data/pretrain/polymers.txt \\
            --emit-sentencepiece-vocab
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # the package is not pip-installed yet
    sys.path.insert(0, str(_SRC))

from polyt5.tokenization import (  # noqa: E402
    PolyT5Tokenizer,
    TokenizerArtifact,
    build_default_vocab,
    build_vocab_from_corpus,
)

DEFAULT_OUT = Path("artifacts/tokenizer/polyt5_vocab.json")


def _read_corpus(path: Path) -> Iterator[str]:
    """Yield non-empty lines from a PSELFIES corpus file.

    Args:
        path: A ``.txt``/``.smi`` file with one PSELFIES string per line.

    Yields:
        Stripped, non-empty lines.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield stripped


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="build_tokenizer",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"destination JSON artifact (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="optional .txt/.smi file of PSELFIES strings, one per line; the base block is "
        "then ranked by frequency (ties broken by token asc) instead of being derived "
        "from the selfies alphabet",
    )
    parser.add_argument(
        "--base-size", type=int, default=199, help="size of the base SELFIES block (paper: 199)"
    )
    parser.add_argument(
        "--sentinels", type=int, default=100, help="number of sentinel tokens (paper: 100)"
    )
    parser.add_argument(
        "--conditioning-size",
        type=int,
        default=154,
        help="size of the conditioning block (paper: 154)",
    )
    parser.add_argument(
        "--emit-sentencepiece-vocab",
        action="store_true",
        help="also write a plain '<token>\\t0' .vocab side file for SentencePiece interop; "
        "the tokenizer itself never reads it and never imports sentencepiece",
    )
    return parser.parse_args(argv)


def _format_summary(artifact: TokenizerArtifact, tokenizer: PolyT5Tokenizer) -> str:
    """Render the composition summary table.

    Args:
        artifact: The freshly built artifact.
        tokenizer: A tokenizer over the same token list.

    Returns:
        A multi-line, printable report.
    """
    meta = artifact.metadata
    groups = meta["group_counts"]
    ranges = meta["id_ranges"]
    provenance = meta["provenance"]

    lines = ["", "Composition", "-" * 62]
    lines.append(f"{'group':<16}{'count':>7}  {'ids':<14}{'provenance':<12}")
    lines.append("-" * 62)
    for name, count in groups.items():
        lo, hi = ranges[name]
        lines.append(f"{name:<16}{count:>7}  {f'{lo}..{hi}':<14}{provenance[name]:<12}")
    lines.append("-" * 62)
    lines.append(f"{'TOTAL':<16}{len(artifact.tokens):>7}")

    base = meta["base_alphabet"]
    cond = meta["conditioning_block"]
    lines += [
        "",
        "Adjustments",
        "-" * 62,
        f"base_selfies   source={base['source']} natural={base['natural_size']} "
        f"padded={base['padded']} dropped={base['dropped']}",
        f"conditioning   natural={cond['natural_size']} padded={cond['padded']} "
        f"trimmed={cond['trimmed']}",
        "",
        "Conditioning groups",
        "-" * 62,
    ]
    for name, size in cond["group_sizes"].items():
        lines.append(f"  {name:<20}{size:>5}")

    lines += [
        "",
        f"selfies version : {meta['selfies_version']}",
        f"artifact version: {artifact.version}",
        f"sha256          : {tokenizer.sha256}",
        "",
        "Reproduction note",
        "-" * 62,
        meta["reproduction_note"],
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Build and write the tokenizer artifact.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args(argv)

    if args.corpus is not None:
        if not args.corpus.exists():
            print(f"error: corpus not found: {args.corpus}", file=sys.stderr)
            return 2
        artifact = build_vocab_from_corpus(
            _read_corpus(args.corpus),
            target_base_size=args.base_size,
            sentinels=args.sentinels,
            conditioning_size=args.conditioning_size,
        )
        artifact.metadata["corpus_path"] = str(args.corpus)
    else:
        artifact = build_default_vocab(
            base_size=args.base_size,
            sentinels=args.sentinels,
            conditioning_size=args.conditioning_size,
        )

    tokenizer = PolyT5Tokenizer.from_artifact(artifact)
    out_path = tokenizer.save(args.out)
    print(_format_summary(artifact, tokenizer))
    print(f"wrote {out_path}")

    if args.emit_sentencepiece_vocab:
        vocab_path = tokenizer.save_sentencepiece_vocab(out_path.with_suffix(".vocab"))
        print(f"wrote {vocab_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
