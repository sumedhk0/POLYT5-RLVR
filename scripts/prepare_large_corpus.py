"""Parallel, resumable, streaming PSMILES -> pre-tokenized corpus builder.

Why this script exists instead of ``scripts/prepare_pretraining_data.py``:

* That script is single-threaded. Measured on this machine it converts
  PSMILES -> ``[At]`` -> PSELFIES at 6,334 rows/s, so 100M rows is 4.4 hours on
  one core while 19 others idle.
* It accumulates every kept PSELFIES string in a Python list. 100M strings do
  not fit in RAM.
* It writes text, which then has to be re-tokenized on every ``__getitem__`` of
  every epoch of every run.

This script fans the chemistry out across a process pool and streams the result
straight into a :class:`~polyt5.data.tokenized_corpus.TokenizedCorpusWriter`, so
memory stays bounded regardless of corpus size::

    raw PSMILES line
        -> star_to_at                      (worker)
        -> validate_psmiles                (worker, RDKit -- the expensive step)
        -> psmiles_to_pselfies             (worker)
        -> tokenizer.encode                (worker, once and forever)
        -> length filter                   (worker)
        -> deduplication                   (PARENT: needs a global view)
        -> TokenizedCorpusWriter.add_packed(PARENT: single writer, ordered .bin)

Usage:
    python scripts/prepare_large_corpus.py \
        --input C:/Users/sumedh/polyt5-data/external/generated_polymer_smiles_dev.txt \
        --output-prefix C:/Users/sumedh/polyt5-data/processed/polyone_dev/corpus \
        --tokenizer artifacts/tokenizer/polyt5_vocab.json

Writes next to ``--output-prefix``: ``corpus.bin`` / ``corpus.idx`` /
``corpus.json`` (the corpus), ``corpus.dedup.u64`` (resume-safe dedup hashes),
``corpus.progress.json`` (resume checkpoint), ``stats.json`` and ``splits.json``.

Windows note: the pool uses the spawn start method, so every heavyweight import
and all top-level work is guarded by ``if __name__ == "__main__"`` and the
tokenizer is built once per worker in a ``Pool`` initializer rather than pickled
with every task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from polyt5.data.prepare import PreparationStats  # noqa: E402
from polyt5.data.splits import save_splits  # noqa: E402
from polyt5.data.tokenized_corpus import (  # noqa: E402
    TokenizedCorpus,
    TokenizedCorpusWriter,
    dedup_sidecar_path,
    verify_corpus,
)

#: Schema version of the resume checkpoint sidecar.
PROGRESS_VERSION = "1.0.0"

#: Splits larger than this are written as .npy sidecars, not inline JSON.
#: 100M indices inline is ~900 MB of text and minutes to parse.
INLINE_SPLIT_LIMIT = 1_000_000

#: Above this many kept sequences, `random.shuffle` over a Python list of ints
#: costs several GB and over a minute; use a numpy permutation instead.
NUMPY_SPLIT_THRESHOLD = 25_000_000

# Per-worker state, populated by _init_worker. Module-level because a spawned
# worker re-imports this module and must not re-read the tokenizer per task.
_WORKER: dict[str, Any] = {}


# --------------------------------------------------------------------------
# worker side
# --------------------------------------------------------------------------


def _init_worker(tokenizer_path: str, max_length: int, dedup: str) -> None:
    """Pool initializer: build the tokenizer once per worker process.

    Args:
        tokenizer_path: Path to the tokenizer JSON artifact.
        max_length: Token budget; rows longer than this are dropped.
        dedup: One of ``"none"``, ``"exact"``, ``"canonical"``.
    """
    from polyt5.chemistry import psmiles_to_pselfies, star_to_at, validate_psmiles
    from polyt5.tokenization import PolyT5Tokenizer

    _WORKER["tokenizer"] = PolyT5Tokenizer.from_file(tokenizer_path)
    _WORKER["star_to_at"] = star_to_at
    _WORKER["validate_psmiles"] = validate_psmiles
    _WORKER["psmiles_to_pselfies"] = psmiles_to_pselfies
    _WORKER["max_length"] = int(max_length)
    _WORKER["dedup"] = dedup


def _hash64(text: str) -> int:
    """Hash a string to a 64-bit int for deduplication.

    Full strings are not kept: 100M canonical PSMILES would cost tens of GB in
    the parent, while 100M 64-bit hashes cost a few. The false-duplicate rate at
    that scale is ~2.7e-4 sequences in expectation (birthday bound over 2^64),
    i.e. far below the corpus's own attrition noise.

    Args:
        text: String to hash.

    Returns:
        A 64-bit unsigned integer.
    """
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def _process_chunk(chunk: list[str]) -> dict[str, Any]:
    """Convert and tokenize one chunk of raw PSMILES lines.

    Runs in a pool worker. Everything except deduplication happens here;
    deduplication needs a corpus-global view and stays in the parent.

    Args:
        chunk: Raw lines exactly as read from the input file.

    Returns:
        A dict with the packed ids (``flat``/``lengths`` as raw bytes), the
        per-row dedup ``hashes``, the raw line count, and the attrition counters
        for the buckets this worker can decide.
    """
    tokenizer = _WORKER["tokenizer"]
    star_to_at = _WORKER["star_to_at"]
    validate_psmiles = _WORKER["validate_psmiles"]
    psmiles_to_pselfies = _WORKER["psmiles_to_pselfies"]
    max_length = _WORKER["max_length"]
    dedup = _WORKER["dedup"]

    flat: list[int] = []
    lengths: list[int] = []
    hashes: list[int] = []
    n_input = 0
    n_parse_failed = 0
    n_wrong_termini = 0
    n_selfies_failed = 0
    n_too_long = 0

    for raw in chunk:
        line = raw.strip()
        if not line:
            continue  # blank padding lines are not rows and are not counted
        n_input += 1

        at_form = star_to_at(line)
        validity = validate_psmiles(at_form)
        if not validity.valid:
            n_parse_failed += 1
            continue
        if not validity.correct_termini:
            n_wrong_termini += 1
            continue

        pselfies = psmiles_to_pselfies(at_form)
        if pselfies is None:
            n_selfies_failed += 1
            continue

        # One encode, not two: the length filter and the stored ids are the
        # same call, so the budget is measured on exactly what the model sees
        # (including EOS), matching prepare._count_tokens with a tokenizer.
        ids = tokenizer.encode(pselfies, add_eos=True, truncation=False)
        if len(ids) > max_length:
            n_too_long += 1
            continue

        flat.extend(ids)
        lengths.append(len(ids))
        if dedup == "canonical":
            hashes.append(_hash64(validity.canonical_psmiles or at_form))
        elif dedup == "exact":
            hashes.append(_hash64(line))

    return {
        "n_raw": len(chunk),
        "n_input": n_input,
        "n_parse_failed": n_parse_failed,
        "n_wrong_termini": n_wrong_termini,
        "n_selfies_failed": n_selfies_failed,
        "n_too_long": n_too_long,
        "flat": np.asarray(flat, dtype=np.uint32).tobytes(),
        "lengths": np.asarray(lengths, dtype=np.int64).tobytes(),
        "hashes": np.asarray(hashes, dtype=np.uint64).tobytes(),
    }


# --------------------------------------------------------------------------
# parent side: input streaming
# --------------------------------------------------------------------------


def _iter_chunks(
    inputs: list[Path],
    chunk_size: int,
    *,
    skip_lines: int = 0,
    limit: int | None = None,
) -> Iterator[list[str]]:
    """Stream the concatenated inputs as fixed-size lists of raw lines.

    Args:
        inputs: Input files, read in the given order (that order is part of the
            corpus identity and of the resume checkpoint).
        chunk_size: Lines per chunk.
        skip_lines: Raw lines to discard first (resume).
        limit: Stop after this many raw lines have been *yielded past* the skip.

    Yields:
        Lists of raw lines, the last one possibly short.
    """
    remaining_skip = int(skip_lines)
    emitted = 0
    for path in inputs:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            if remaining_skip:
                consumed = sum(1 for _ in islice(fh, remaining_skip))
                remaining_skip -= consumed
                if remaining_skip:
                    continue  # this whole file was already processed
            while True:
                take = chunk_size
                if limit is not None:
                    take = min(take, limit - emitted)
                    if take <= 0:
                        return
                chunk = list(islice(fh, take))
                if not chunk:
                    break
                emitted += len(chunk)
                yield chunk


def _throttle(chunks: Iterator[list[str]], gate: threading.Semaphore) -> Iterator[list[str]]:
    """Bound how many chunks may be in flight inside the pool.

    ``Pool.imap`` drains its input iterable as fast as the task handler thread
    can run, which for an 8 GB file means reading the whole file into the task
    queue. Acquiring a semaphore before each yield -- released by the parent as
    each result lands -- keeps memory flat.

    Args:
        chunks: The underlying chunk stream.
        gate: Semaphore whose initial value is the in-flight budget.

    Yields:
        The same chunks, throttled.
    """
    for chunk in chunks:
        gate.acquire()
        yield chunk


# --------------------------------------------------------------------------
# parent side: resume checkpoint
# --------------------------------------------------------------------------


def _progress_path(prefix: Path) -> Path:
    """Return the resume-checkpoint path for a corpus prefix."""
    return prefix.with_name(prefix.name + ".progress.json")


def _fingerprint(args: argparse.Namespace, tokenizer_sha256: str) -> dict[str, Any]:
    """Describe the build settings that a resume must not change.

    Args:
        args: Parsed CLI arguments.
        tokenizer_sha256: Vocabulary identity.

    Returns:
        A JSON-serializable fingerprint dict.
    """
    return {
        "inputs": [str(Path(p).resolve()) for p in args.input],
        "chunk_size": int(args.chunk_size),
        "max_length": int(args.max_length),
        "dedup": args.dedup,
        "tokenizer_sha256": tokenizer_sha256,
    }


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write the resume checkpoint.

    Args:
        path: Checkpoint path.
        payload: Checkpoint contents.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_dedup_hashes(path: Path, n_sequences: int) -> set[int]:
    """Reload (and repair) the deduplication hash sidecar for a resume.

    The sidecar holds exactly one uint64 per kept sequence, so truncating it to
    ``n_sequences`` puts it back in lockstep with the corpus after the writer
    rewinds a torn write.

    Args:
        path: Sidecar path.
        n_sequences: Sequences the corpus was rewound to.

    Returns:
        The set of hashes for the sequences that survive the rewind.
    """
    if not path.exists() or n_sequences == 0:
        if path.exists():
            with path.open("r+b") as fh:
                fh.truncate(0)
        return set()
    hashes = np.fromfile(path, dtype=np.uint64)
    if hashes.size > n_sequences:
        hashes = hashes[:n_sequences]
        with path.open("r+b") as fh:
            fh.truncate(n_sequences * 8)
    return set(hashes.tolist())


# --------------------------------------------------------------------------
# parent side: splits
# --------------------------------------------------------------------------


def _make_splits(
    n: int, fractions: list[float], seed: int, method: str
) -> dict[str, np.ndarray]:
    """Partition ``range(n)`` into train/val/test index arrays.

    Args:
        n: Corpus size.
        fractions: ``[train, val, test]``, summing to 1.
        seed: RNG seed.
        method: ``"python"`` (``polyt5.data.splits``), ``"numpy"``, or ``"auto"``.

    Returns:
        ``{"train": ..., "val": ..., "test": ...}``.

    Note:
        The two methods draw different permutations, so a corpus split with
        ``numpy`` is not index-comparable with one split by ``python``. The
        chosen method is recorded in ``splits.json`` for exactly that reason.
        ``polyt5.data.splits.random_split`` is the project's canonical splitter
        and is used whenever it is affordable; above ~25M sequences its Python
        list of ints costs several GB and over a minute, which is why the numpy
        path exists at all.
    """
    if method == "auto":
        method = "numpy" if n > NUMPY_SPLIT_THRESHOLD else "python"

    if method == "python":
        from polyt5.data.splits import make_pretraining_splits

        parts = make_pretraining_splits(
            n, seed=seed, train=fractions[0], val=fractions[1], test=fractions[2]
        )
        return {k: np.asarray(v, dtype=np.int64) for k, v in parts.items()}

    if method != "numpy":
        raise ValueError(f"unknown split method {method!r}")

    perm = np.random.default_rng(seed).permutation(n)
    # Same cumulative-rounded boundaries as polyt5.data.splits.random_split.
    bounds = []
    cumulative = 0.0
    for fraction in fractions:
        cumulative += fraction
        bounds.append(round(cumulative * n))
    bounds[-1] = n
    start = 0
    out: dict[str, np.ndarray] = {}
    for name, end in zip(("train", "val", "test"), bounds, strict=True):
        out[name] = perm[start:end]
        start = end
    return out


def _write_splits(
    prefix: Path,
    splits: dict[str, np.ndarray],
    *,
    n: int,
    seed: int,
    method: str,
    fmt: str,
) -> Path:
    """Write ``splits.json`` beside the corpus, spilling big splits to ``.npy``.

    Args:
        prefix: Corpus path prefix.
        splits: Split name -> index array.
        n: Corpus size.
        seed: Seed used.
        method: Split method actually used.
        fmt: ``"auto"``, ``"inline"`` or ``"npy"``.

    Returns:
        The path of ``splits.json``.
    """
    if fmt == "auto":
        fmt = "npy" if n > INLINE_SPLIT_LIMIT else "inline"

    payload: dict[str, Any] = {
        "seed": seed,
        "n": n,
        "split_method": method,
        "format": fmt,
        "split_sizes": {name: int(idx.size) for name, idx in splits.items()},
        "fractions": {name: (int(idx.size) / n if n else 0.0) for name, idx in splits.items()},
    }
    for name, idx in splits.items():
        if fmt == "npy":
            sidecar = prefix.with_name(f"{prefix.name}.split_{name}.npy")
            np.save(sidecar, idx.astype(np.int64, copy=False))
            payload[name] = {"npy": sidecar.name, "size": int(idx.size)}
        else:
            payload[name] = idx.astype(np.int64, copy=False).tolist()
    return save_splits(prefix.parent / "splits.json", payload)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Argument list, or None for ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", action="append", required=True, type=Path,
                        help="Raw PSMILES text, one per line. Repeat for several files; "
                             "the order is part of the corpus identity.")
    parser.add_argument("--output-prefix", required=True, type=Path,
                        help="Corpus path prefix, e.g. .../processed/polyone/corpus")
    parser.add_argument("--tokenizer", type=Path,
                        default=REPO_ROOT / "artifacts" / "tokenizer" / "polyt5_vocab.json")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                        help="Pool size (default: cpu_count - 2, leaving room for the writer).")
    parser.add_argument("--chunk-size", type=int, default=50_000,
                        help="Raw lines per pool task.")
    parser.add_argument("--max-length", type=int, default=200,
                        help="Token budget including EOS (paper: 200).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Debug: stop after this many raw input lines.")
    parser.add_argument("--dedup", choices=("none", "exact", "canonical"), default="exact",
                        help="Deduplication key; see the note in the module docstring.")
    parser.add_argument("--splits", nargs=3, type=float, default=[0.9, 0.05, 0.05],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--split-method", choices=("auto", "python", "numpy"), default="auto")
    parser.add_argument("--splits-format", choices=("auto", "inline", "npy"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="Continue from the .progress.json checkpoint if one exists.")
    parser.add_argument("--report-every", type=int, default=20,
                        help="Chunks between rate lines on stdout.")
    parser.add_argument("--dtype", choices=("uint16", "uint32"), default="uint16")
    return parser.parse_args(argv)


def _format_eta(seconds: float) -> str:
    """Render a duration as ``HH:MM:SS``.

    Args:
        seconds: Duration in seconds.

    Returns:
        The formatted string, or ``"--:--:--"`` when unknown.
    """
    if seconds < 0 or seconds != seconds or seconds == float("inf"):
        return "--:--:--"
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list, or None for ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    args = parse_args(argv)

    inputs = [Path(p) for p in args.input]
    missing = [p for p in inputs if not p.exists()]
    if missing:
        print(f"ERROR: input file(s) not found: {', '.join(str(p) for p in missing)}",
              file=sys.stderr)
        return 2

    from polyt5.tokenization import PolyT5Tokenizer

    tokenizer = PolyT5Tokenizer.from_file(args.tokenizer)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    fingerprint = _fingerprint(args, tokenizer.sha256)
    progress_path = _progress_path(prefix)
    dedup_path = dedup_sidecar_path(prefix)

    # ---------------------------------------------------------------- resume
    resume_state: dict[str, Any] | None = None
    if args.resume and progress_path.exists():
        resume_state = json.loads(progress_path.read_text(encoding="utf-8"))
        if resume_state.get("fingerprint") != fingerprint:
            print("ERROR: --resume refused: the build settings changed since the checkpoint.\n"
                  f"  checkpoint: {json.dumps(resume_state.get('fingerprint'), indent=2)}\n"
                  f"  now:        {json.dumps(fingerprint, indent=2)}", file=sys.stderr)
            return 2
    elif args.resume:
        print(f"note: --resume given but no checkpoint at {progress_path}; starting fresh")

    if resume_state is not None:
        skip_lines = int(resume_state["n_lines_consumed"])
        done_sequences = int(resume_state["n_sequences"])
        totals = dict(resume_state["stats"])
        chunk_index = int(resume_state["n_chunks_done"])
        print(f"resuming: {skip_lines:,} input lines already consumed, "
              f"{done_sequences:,} sequences on disk")
    else:
        skip_lines = 0
        done_sequences = 0
        totals = PreparationStats().to_dict()
        chunk_index = 0
        if dedup_path.exists():
            dedup_path.unlink()

    seen: set[int] = set()
    if args.dedup != "none":
        seen = _load_dedup_hashes(dedup_path, done_sequences)
        if seen:
            print(f"reloaded {len(seen):,} deduplication hashes")

    writer = TokenizedCorpusWriter(
        prefix,
        tokenizer_sha256=tokenizer.sha256,
        tokenizer_path=str(args.tokenizer),
        max_length=args.max_length,
        dtype=args.dtype,
        vocab_size=tokenizer.vocab_size,
        source_path=[str(p) for p in inputs],
        append=resume_state is not None,
        n_sequences=done_sequences if resume_state is not None else None,
    )
    writer.set_metadata(
        dedup=args.dedup,
        length_filter_unit="tokenizer_ids_including_eos",
        workers=int(args.workers),
        chunk_size=int(args.chunk_size),
        seed=int(args.seed),
    )
    dedup_fh = dedup_path.open("ab") if args.dedup != "none" else None

    print(f"tokenizer:  {args.tokenizer} (vocab={tokenizer.vocab_size}, "
          f"sha256={tokenizer.sha256[:16]})")
    print(f"inputs:     {', '.join(str(p) for p in inputs)}")
    print(f"output:     {prefix}")
    print(f"workers:    {args.workers}   chunk-size: {args.chunk_size}   dedup: {args.dedup}")

    n_lines_consumed = skip_lines
    started = time.time()
    interrupted = False
    progress_bar = None
    if sys.stderr.isatty():
        try:
            from tqdm import tqdm

            progress_bar = tqdm(desc="prepare", unit="row", unit_scale=True, file=sys.stderr)
        except ImportError:  # pragma: no cover - tqdm is a hard dependency
            progress_bar = None

    def checkpoint() -> None:
        """Flush the corpus, then record a resume point that cannot over-claim."""
        if dedup_fh is not None:
            dedup_fh.flush()
        n_seq, n_tok = writer.flush()
        _write_progress(progress_path, {
            "format_version": PROGRESS_VERSION,
            "fingerprint": fingerprint,
            "n_chunks_done": chunk_index,
            "n_lines_consumed": n_lines_consumed,
            "n_sequences": n_seq,
            "n_tokens": n_tok,
            "stats": totals,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })

    gate = threading.Semaphore(2 * int(args.workers) + 2)
    chunks = _throttle(
        _iter_chunks(inputs, args.chunk_size, skip_lines=skip_lines, limit=args.limit),
        gate,
    )

    pool = mp.Pool(
        processes=int(args.workers),
        initializer=_init_worker,
        initargs=(str(args.tokenizer), int(args.max_length), args.dedup),
    )
    try:
        for result in pool.imap(_process_chunk, chunks, chunksize=1):
            gate.release()
            chunk_index += 1
            n_lines_consumed += int(result["n_raw"])

            totals["n_input"] += result["n_input"]
            totals["n_parse_failed"] += result["n_parse_failed"]
            totals["n_wrong_termini"] += result["n_wrong_termini"]
            totals["n_selfies_failed"] += result["n_selfies_failed"]
            totals["n_too_long"] += result["n_too_long"]

            flat = np.frombuffer(result["flat"], dtype=np.uint32)
            lengths = np.frombuffer(result["lengths"], dtype=np.int64)

            if args.dedup != "none" and lengths.size:
                hashes = np.frombuffer(result["hashes"], dtype=np.uint64)
                keep = np.ones(lengths.size, dtype=bool)
                for i, value in enumerate(hashes.tolist()):
                    if value in seen:
                        keep[i] = False
                    else:
                        seen.add(value)
                n_dupes = int((~keep).sum())
                if n_dupes:
                    totals["n_duplicate"] += n_dupes
                    flat = flat[np.repeat(keep, lengths)]
                    lengths = lengths[keep]
                    hashes = hashes[keep]
                if dedup_fh is not None and hashes.size:
                    dedup_fh.write(hashes.astype(np.uint64, copy=False).tobytes())

            if lengths.size:
                totals["n_kept"] += int(lengths.size)
                writer.add_packed(flat, lengths)

            if progress_bar is not None:
                progress_bar.update(int(result["n_raw"]))
            if args.report_every and chunk_index % args.report_every == 0:
                elapsed = max(time.time() - started, 1e-9)
                done_now = n_lines_consumed - skip_lines
                rate = done_now / elapsed
                keep_rate = totals["n_kept"] / totals["n_input"] if totals["n_input"] else 0.0
                print(f"[{_format_eta(elapsed)}] {n_lines_consumed:,} rows  "
                      f"{rate:,.0f} rows/s  kept {keep_rate:.1%}  "
                      f"seq {writer.n_sequences:,}  tok {writer.n_tokens:,}", flush=True)
                checkpoint()
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted: checkpointing so --resume can continue", file=sys.stderr)
        pool.terminate()
    finally:
        if not interrupted:
            pool.close()
        pool.join()
        if progress_bar is not None:
            progress_bar.close()

    checkpoint()
    if dedup_fh is not None:
        dedup_fh.close()

    stats = PreparationStats(**totals)
    writer.set_preparation_stats(stats)
    metadata = writer.close()

    elapsed = time.time() - started
    processed = n_lines_consumed - skip_lines
    print(f"\nprocessed {processed:,} lines in {_format_eta(elapsed)} "
          f"({processed / max(elapsed, 1e-9):,.0f} rows/s)")
    print("attrition:", json.dumps(stats.to_dict()))

    if interrupted:
        print("stopped early; rerun with --resume to finish. "
              "splits.json/stats.json NOT written for a partial corpus.", file=sys.stderr)
        return 130

    if stats.n_kept == 0:
        print("ERROR: no rows survived preparation; corpus is empty.", file=sys.stderr)
        return 1

    # --------------------------------------------------------------- splits
    method = args.split_method
    if method == "auto":
        method = "numpy" if metadata["n_sequences"] > NUMPY_SPLIT_THRESHOLD else "python"
    splits = _make_splits(metadata["n_sequences"], list(args.splits), args.seed, method)
    splits_path = _write_splits(
        prefix, splits, n=metadata["n_sequences"], seed=args.seed,
        method=method, fmt=args.splits_format,
    )
    print(f"wrote {splits_path}  "
          f"({', '.join(f'{k}={v.size:,}' for k, v in splits.items())}, method={method})")

    # ---------------------------------------------------------------- stats
    report = verify_corpus(prefix, tokenizer)
    stats_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(p) for p in inputs],
        "output_prefix": str(prefix),
        "limit": args.limit,
        "max_length": args.max_length,
        "dedup": args.dedup,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "tokenizer_path": str(args.tokenizer),
        "tokenizer_sha256": tokenizer.sha256,
        "length_filter_unit": "tokenizer_ids_including_eos",
        "attrition": stats.to_dict(),
        "split_sizes": {k: int(v.size) for k, v in splits.items()},
        "split_method": method,
        "elapsed_seconds": round(elapsed, 3),
        "rows_per_second": round(processed / max(elapsed, 1e-9), 1),
        "verify": report,
    }
    (prefix.parent / "stats.json").write_text(
        json.dumps(stats_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {prefix.parent / 'stats.json'}")

    corpus = TokenizedCorpus.from_prefix(prefix)
    bytes_on_disk = sum(
        p.stat().st_size for p in prefix.parent.glob(prefix.name + ".*") if p.is_file()
    )
    keep_rate = stats.n_kept / stats.n_input if stats.n_input else 0.0
    print(f"corpus:   {len(corpus):,} sequences, {corpus.n_tokens:,} tokens, "
          f"mean length {report['mean_length']:.1f}")
    print(f"on disk:  {bytes_on_disk / 1e9:.3f} GB")
    print(f"done: kept {stats.n_kept:,}/{stats.n_input:,} rows ({keep_rate:.2%})")
    return 0


if __name__ == "__main__":
    # Windows/macOS use spawn: every worker re-imports this module, so nothing
    # heavy may run at import time and the pool must be created from here.
    mp.freeze_support()
    raise SystemExit(main())
