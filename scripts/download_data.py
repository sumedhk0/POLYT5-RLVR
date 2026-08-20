"""Download registered external datasets with provenance sidecars.

Usage:
    python scripts/download_data.py --dataset pi1m --dest data/external
    python scripts/download_data.py --dataset all --dest data/external
    python scripts/download_data.py --dataset polyone_dev --yes-large

Sources flagged large (the 9 GB polyOne corpus) are refused without
``--yes-large``; the refusal prints the size first. Every fetched (or adopted)
file gets a ``<file>.provenance.json`` sidecar with URL, bytes and SHA-256.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from polyt5.data.sources import (  # noqa: E402
    SOURCES,
    ConfirmationRequiredError,
    download,
)


def _human_bytes(n: int) -> str:
    """Format a byte count for humans, e.g. ``8.14 GB``."""
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= scale:
            return f"{n / scale:.2f} {unit}"
    return f"{n} B"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        choices=[*SOURCES, "all"],
        required=True,
        help="which registered dataset to fetch ('all' skips large ones unless --yes-large)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=REPO_ROOT / "data" / "external",
        help="destination directory (default: data/external)",
    )
    parser.add_argument(
        "--yes-large",
        action="store_true",
        help="explicitly approve sources flagged as large downloads (polyOne, ~9 GB)",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if the file already exists"
    )
    args = parser.parse_args(argv)

    names = list(SOURCES) if args.dataset == "all" else [args.dataset]
    exit_code = 0
    for name in names:
        source = SOURCES[name]
        print(f"\n=== {source.name} ===")
        print(f"  {source.description}")
        print(f"  url:     {source.url}")
        print(f"  size:    ~{_human_bytes(source.approx_bytes)}")
        print(f"  license: {source.license}")
        try:
            record = download(
                source, args.dest, confirm_large=args.yes_large, force=args.force
            )
        except ConfirmationRequiredError as exc:
            print(f"  SKIPPED: {exc}")
            if args.dataset != "all":
                exit_code = 1  # an explicitly requested dataset was refused
            continue
        print(f"  path:    {record.path}")
        print(f"  bytes:   {record.bytes}")
        print(f"  sha256:  {record.sha256}")
        print(f"  sidecar: {record.path}.provenance.json")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
