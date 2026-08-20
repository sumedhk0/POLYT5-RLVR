"""Build a :class:`~polyt5.chemistry.scalable_novelty.ScalableNoveltyIndex`.

Three input routes, mixable in one run:

* ``--from-corpus PREFIX`` -- the cheap one. Reads the ``<prefix>.dedup.u64``
  sidecar that ``scripts/prepare_large_corpus.py`` already wrote, so a 100M-row
  index costs an 800 MB read and a sort instead of re-canonicalizing the corpus.
* ``--from-text PATH`` -- one string per line, repeatable.
* ``--from-jsonl PATH --field target`` -- one JSON object per line.

Input is streamed and hashed in chunks: the strings never accumulate, so peak
memory is one chunk plus 8 bytes per kept row (800 MB at 100M rows).

Example::

    python scripts/build_novelty_index.py \\
        --from-corpus ~/polyt5-data/processed/polyone/corpus \\
        --out artifacts/novelty/polyone

    python scripts/build_novelty_index.py \\
        --from-text data/pi1m.txt --hash-space canonical_psmiles --canonicalize \\
        --workers 8 --out artifacts/novelty/pi1m

Writes ``<out>.u64`` (sorted, deduplicated uint64) and ``<out>.json``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from polyt5.chemistry.conversion import pselfies_to_psmiles  # noqa: E402
from polyt5.chemistry.scalable_novelty import (  # noqa: E402
    HASH_SPACES,
    INDEX_DTYPE,
    NoveltyIndexMetadata,
    ScalableNoveltyIndex,
    hash64_for_space,
)

#: Strings hashed per chunk. Large enough to amortise the pool round trip,
#: small enough that a chunk of PSMILES is a few MB.
CHUNK_SIZE = 20_000

#: Per-worker state, populated by :func:`_init_worker`.
_WORKER: dict[str, Any] = {}


def _init_worker(hash_space: str, canonicalize: bool) -> None:
    """Pool initializer: remember how this run hashes.

    Args:
        hash_space: One of :data:`polyt5.chemistry.scalable_novelty.HASH_SPACES`.
        canonicalize: Canonicalize PSMILES before hashing.
    """
    _WORKER["hash_space"] = hash_space
    _WORKER["canonicalize"] = canonicalize


def _hash_chunk(chunk: list[str]) -> bytes:
    """Hash one chunk of strings in a worker.

    Returns raw bytes rather than an array so the pickled result is exactly the
    payload, with no dtype metadata per chunk.

    Args:
        chunk: Strings to hash.

    Returns:
        The kept hashes as ``uint64`` bytes.
    """
    codes = hash64_for_space(
        chunk,
        hash_space=_WORKER["hash_space"],
        canonicalize=_WORKER["canonicalize"] or None,
    )
    return codes.tobytes()


def _chunked(items: Iterable[str], size: int) -> Iterator[list[str]]:
    """Yield lists of at most ``size`` items from a stream.

    Args:
        items: Any iterable of strings.
        size: Maximum chunk length.

    Yields:
        Non-empty lists of strings.
    """
    iterator = iter(items)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            return
        yield chunk


def _read_text(path: Path) -> Iterator[str]:
    """Stream non-empty stripped lines from a text file.

    Args:
        path: File to read.

    Yields:
        One string per line.
    """
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                yield line


def _read_jsonl(path: Path, field: str) -> Iterator[str]:
    """Stream one field from a JSON-lines file.

    Rows that are not objects, lack the field, or do not parse are skipped:
    a malformed row is not a polymer, and a build over 100M rows must not die
    on one of them.

    Args:
        path: File to read.
        field: Key whose value is the polymer string.

    Yields:
        One string per usable row.
    """
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                value = row.get(field)
                if isinstance(value, str) and value.strip():
                    yield value.strip()


def _limited(items: Iterable[str], limit: int | None) -> Iterator[str]:
    """Yield at most ``limit`` items.

    Args:
        items: Source stream.
        limit: Maximum count, or ``None`` for all.

    Yields:
        The (possibly truncated) stream.
    """
    if limit is None:
        yield from items
        return
    yield from islice(items, limit)


def _hash_stream(
    strings: Iterable[str], *, hash_space: str, canonicalize: bool, workers: int
) -> tuple[list[np.ndarray], int, int]:
    """Hash a stream of strings, in parallel when asked.

    Args:
        strings: Source stream; consumed once, never materialised.
        hash_space: Space to hash in.
        canonicalize: Canonicalize PSMILES before hashing.
        workers: Worker processes; 1 hashes inline.

    Returns:
        A ``(blocks, n_read, n_hashed)`` triple: the per-chunk ``uint64``
        arrays, the number of strings read, and the number that hashed.
    """
    blocks: list[np.ndarray] = []
    n_read = 0
    n_hashed = 0
    started = time.perf_counter()
    last_report = started

    def _absorb(payload: bytes | np.ndarray, chunk_len: int) -> None:
        nonlocal n_read, n_hashed, last_report
        codes = (
            np.frombuffer(payload, dtype=INDEX_DTYPE)
            if isinstance(payload, bytes)
            else payload
        )
        if codes.size:
            blocks.append(np.array(codes, dtype=INDEX_DTYPE, copy=True))
        n_read += chunk_len
        n_hashed += int(codes.size)
        now = time.perf_counter()
        if now - last_report >= 5.0:
            rate = n_read / max(now - started, 1e-9)
            print(f"  {n_read:,} read  {n_hashed:,} hashed  {rate:,.0f} rows/s", flush=True)
            last_report = now

    chunks = _chunked(strings, CHUNK_SIZE)
    if workers <= 1:
        _init_worker(hash_space, canonicalize)
        for chunk in chunks:
            _absorb(_hash_chunk(chunk), len(chunk))
    else:
        context = mp.get_context("spawn")
        with context.Pool(
            workers, initializer=_init_worker, initargs=(hash_space, canonicalize)
        ) as pool:
            pending: list[tuple[Any, int]] = []
            for chunk in chunks:
                pending.append((pool.apply_async(_hash_chunk, (chunk,)), len(chunk)))
                # Bounded queue: never let the reader run ahead of the workers,
                # or a 100M-row file becomes 100M queued strings.
                while len(pending) >= workers * 4:
                    result, size = pending.pop(0)
                    _absorb(result.get(), size)
            for result, size in pending:
                _absorb(result.get(), size)
    return blocks, n_read, n_hashed


def _corpus_hashes(prefix: Path, hash_space: str | None) -> tuple[np.ndarray, str]:
    """Load the deduplication sidecar of a prepared corpus.

    Args:
        prefix: Corpus path prefix.
        hash_space: Asserted space, or ``None`` to read it from the corpus
            metadata.

    Returns:
        The raw hashes and the space they are in.
    """
    index = ScalableNoveltyIndex.from_dedup_sidecar(prefix, hash_space=hash_space)
    return index.hashes, index.hash_space


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Build a scalable novelty index from a corpus, text or JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--from-corpus", type=Path, default=None,
                        help="Prepared corpus prefix; reads its .dedup.u64 sidecar.")
    parser.add_argument("--from-text", type=Path, action="append", default=[],
                        help="Text file, one string per line. Repeatable.")
    parser.add_argument("--from-jsonl", type=Path, action="append", default=[],
                        help="JSON-lines file. Repeatable. See --field.")
    parser.add_argument("--field", default="target",
                        help="Field of each JSONL row holding the polymer string.")
    parser.add_argument("--hash-space", choices=HASH_SPACES, default=None,
                        help="String space to hash in. Inferred from the corpus "
                             "metadata when only --from-corpus is used.")
    parser.add_argument("--canonicalize", action="store_true",
                        help="Canonicalize PSMILES before hashing. Requires "
                             "--hash-space canonical_psmiles. Costs an RDKit parse and "
                             "canonical write per string (~10-40 us), i.e. hours for a "
                             "100M-row corpus -- prefer --from-corpus, whose hashes were "
                             "canonicalized once during preparation.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output stem; writes <out>.u64 and <out>.json.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Worker processes for hashing text/JSONL input.")
    parser.add_argument(
        "--max-drop-rate", type=float, default=0.05,
        help="Fail if more than this fraction of input strings cannot be hashed. Guards against "
             "a space mismatch silently producing a near-empty index, where every subsequent "
             "novelty query answers 'novel'.",
    )
    parser.add_argument(
        "--from-pselfies", action="store_true",
        help="Treat --from-text / --from-jsonl input as PSELFIES and convert it to PSMILES "
             "before hashing. Required for corpora whose stored form is PSELFIES, such as the "
             "generation-task JSONL files.",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many input strings (per run, not per file).")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build an index from the command line.

    Args:
        argv: Argument vector; ``None`` uses ``sys.argv``.

    Returns:
        Process exit code.
    """
    args = build_argument_parser().parse_args(argv)

    text_sources = [(path, None) for path in args.from_text]
    jsonl_sources = [(path, args.field) for path in args.from_jsonl]
    string_sources = text_sources + jsonl_sources
    if args.from_corpus is None and not string_sources:
        print("nothing to do: pass --from-corpus, --from-text or --from-jsonl",
              file=sys.stderr)
        return 2
    for path, _ in string_sources:
        if not path.exists():
            print(f"input not found: {path}", file=sys.stderr)
            return 2

    hash_space = args.hash_space
    if string_sources and hash_space is None:
        print("--hash-space is required when hashing text or JSONL input; a wrong "
              "space reports every candidate as novel", file=sys.stderr)
        return 2
    if args.canonicalize and hash_space != "canonical_psmiles":
        print("--canonicalize requires --hash-space canonical_psmiles", file=sys.stderr)
        return 2
    if hash_space == "canonical_psmiles" and string_sources and not args.canonicalize:
        print("--hash-space canonical_psmiles requires --canonicalize when hashing "
              "raw strings, or the stored hashes are not canonical", file=sys.stderr)
        return 2

    started = time.perf_counter()
    blocks: list[np.ndarray] = []
    provenance: list[str] = []
    n_read = 0
    n_hashed = 0

    if args.from_corpus is not None:
        print(f"reading dedup sidecar of {args.from_corpus}", flush=True)
        corpus_hashes, corpus_space = _corpus_hashes(args.from_corpus, hash_space)
        if hash_space is None:
            hash_space = corpus_space
        blocks.append(np.asarray(corpus_hashes, dtype=INDEX_DTYPE))
        n_read += int(corpus_hashes.size)
        n_hashed += int(corpus_hashes.size)
        provenance.append(str(args.from_corpus))
        print(f"  {corpus_hashes.size:,} hashes, space={corpus_space}", flush=True)

    remaining = args.limit
    for path, field in string_sources:
        if remaining is not None and remaining <= 0:
            break
        print(f"hashing {path}", flush=True)
        stream = _read_text(path) if field is None else _read_jsonl(path, field)
        if args.from_pselfies:
            # Stored PSELFIES must become PSMILES before a PSMILES-space hash;
            # canonicalizing a PSELFIES string returns None and the row would be
            # dropped silently. Rows that fail conversion yield "" and are
            # counted as unhashable by the drop-rate guard below.
            stream = (pselfies_to_psmiles(text) or "" for text in stream)
        chunk_blocks, chunk_read, chunk_hashed = _hash_stream(
            _limited(stream, remaining),
            hash_space=str(hash_space),
            canonicalize=bool(args.canonicalize),
            workers=max(1, int(args.workers)),
        )
        blocks.extend(chunk_blocks)
        n_read += chunk_read
        n_hashed += chunk_hashed
        provenance.append(str(path))
        if remaining is not None:
            remaining -= chunk_read

    raw = (
        np.concatenate(blocks) if blocks else np.empty(0, dtype=INDEX_DTYPE)
    )
    metadata = NoveltyIndexMetadata(
        n_entries=0, hash_space=str(hash_space), source=" ".join(provenance)
    )
    index = ScalableNoveltyIndex.from_hashes(raw, metadata=metadata)
    data_path = index.save(args.out)
    elapsed = time.perf_counter() - started

    size_mb = data_path.stat().st_size / 1e6
    rate = n_read / elapsed if elapsed > 0 else float("nan")
    print(
        f"read {n_read:,} rows -> {n_hashed:,} hashed -> {len(index):,} unique\n"
        f"space:   {index.hash_space}\n"
        f"wrote:   {data_path} ({size_mb:,.1f} MB) + {data_path.with_suffix('.json')}\n"
        f"elapsed: {elapsed:,.1f} s ({rate:,.0f} rows/s)",
        flush=True,
    )

    # A silently near-empty index is the worst outcome this script can produce:
    # every later novelty query answers "novel", which reads as a great result
    # rather than as a broken build. The usual cause is a space mismatch --
    # feeding PSELFIES to a canonical_psmiles build, where canonicalization
    # returns None for every row and the rows are quietly discarded.
    dropped = n_read - n_hashed
    drop_rate = dropped / n_read if n_read else 0.0
    if drop_rate > args.max_drop_rate:
        print(
            f"\nERROR: {dropped:,} of {n_read:,} input strings ({drop_rate:.1%}) could not be "
            f"hashed, above the --max-drop-rate of {args.max_drop_rate:.1%}.\n"
            f"       The index was still written, but it is almost certainly wrong.\n"
            f"       Most likely cause: the input is not in {index.hash_space!r} space. "
            f"PSELFIES must be converted with chemistry.pselfies_to_psmiles before a\n"
            f"       canonical_psmiles build -- canonicalizing a PSELFIES string returns None "
            f"and the row is dropped.\n"
            f"       Re-run with --max-drop-rate 1.0 if this attrition is genuinely expected.",
            flush=True,
        )
        return 1
    if dropped:
        print(f"note: {dropped:,} rows ({drop_rate:.2%}) could not be hashed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
