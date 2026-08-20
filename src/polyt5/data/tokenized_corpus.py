"""On-disk, memory-mapped, pre-tokenized PSELFIES corpus.

Why this module exists
----------------------
:class:`polyt5.data.datasets.PSelfiesCorpus` keeps every PSELFIES string in a
Python list and tokenizes inside ``__getitem__``. That is fine at 1M polymers
and impossible at 100M:

* Measured on this project's polyT5-small run, epoch 0 took 5,215 s and
  epoch 1 took 270 s. The 19x gap is per-item Python overhead plus first-touch
  paging, not compute.
* Tokenization is not the bottleneck (77,665 seq/s on one core), so paying it
  once at preparation time and never again is nearly free.
* 100M PSELFIES strings do not fit in RAM at all, and a ``DataLoader`` with
  ``num_workers>0`` would copy that list into every worker.

So the corpus is tokenized once and stored as a flat array of ids that every
worker process memory-maps read-only.

On-disk format
--------------
Three files share one path prefix, e.g. ``.../polyone/corpus``::

    <prefix>.bin    Flat array of token ids, sequences concatenated with no
                    separators, in the ``dtype`` named by the metadata
                    (``uint16`` by default -- the vocabulary is 458 tokens, and
                    the writer refuses ``uint16`` for any vocabulary that does
                    not fit). Little/big endianness is the host's; these are
                    derived artifacts, rebuilt rather than shipped.

    <prefix>.idx    ``uint64`` offsets into ``.bin``, length ``n_sequences+1``,
                    starting at 0 and ending at ``n_tokens``. Sequence ``i`` is
                    ``bin[idx[i]:idx[i+1]]``. Storing offsets rather than
                    lengths makes random access a pure slice and makes a
                    truncated write detectable.

    <prefix>.json   Metadata::

                        format_version   this module's layout version
                        n_sequences      rows in the corpus
                        n_tokens         total ids in <prefix>.bin
                        dtype            "uint16" / "uint32"
                        max_length       token budget applied at build time
                        tokenizer_sha256 vocabulary identity (load-bearing)
                        tokenizer_path   where that vocabulary came from
                        source_path      raw input path(s)
                        created_utc      ISO-8601 UTC timestamp
                        polyt5_version   installed package version
                        preparation_stats  PreparationStats.to_dict()

    <prefix>.dedup.u64   Optional sidecar written by the parallel builder: one
                    ``uint64`` hash per kept sequence, so an interrupted job can
                    rebuild its deduplication set on resume. Not required to
                    read the corpus.

``tokenizer_sha256`` is not decoration. A corpus tokenized with a different
vocabulary is a different corpus -- the same id means a different token -- so
:func:`verify_corpus` raises :class:`TokenizerMismatchError` rather than
returning a flag, and ``scripts/pretrain.py`` refuses to train through it.

Process safety
--------------
``np.memmap`` objects must not survive a ``fork``/``spawn``. :class:`TokenizedCorpus`
therefore opens its maps lazily and keys them by ``os.getpid()``, reopening in
each DataLoader worker, and drops them from ``__getstate__`` so pickling for
Windows spawn never carries a mapping. ``tests/test_tokenized_corpus.py``
asserts that ``num_workers=2`` yields byte-identical sequences to
``num_workers=0``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch.utils.data

__all__ = [
    "CORPUS_FORMAT_VERSION",
    "DTYPES",
    "MemmapPSelfiesDataset",
    "TokenizedCorpus",
    "TokenizedCorpusView",
    "TokenizedCorpusWriter",
    "TokenizerMismatchError",
    "load_split_indices",
    "verify_corpus",
]

#: Layout version of the three-file format. Bump on any breaking change.
CORPUS_FORMAT_VERSION = "1.0.0"

#: Token-id dtypes this format supports, smallest first.
DTYPES: dict[str, type[np.integer]] = {"uint16": np.uint16, "uint32": np.uint32}

#: Offsets are always uint64: 100M sequences x 200 tokens overflows uint32.
OFFSET_DTYPE = np.uint64

#: Python-buffer flush threshold for :meth:`TokenizedCorpusWriter.add`.
_DEFAULT_FLUSH_TOKENS = 1 << 22  # 4M ids ~ 8 MB at uint16


class TokenizerMismatchError(ValueError):
    """Raised when a corpus was built with a different tokenizer vocabulary."""


def _polyt5_version() -> str:
    """Return the installed package version, or ``"unknown"``.

    Returns:
        The distribution version string.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("polyt5-rlvr")
    except PackageNotFoundError:  # pragma: no cover - editable installs always resolve
        return "unknown"
    except Exception:  # pragma: no cover - defensive: never fail a 4-hour job on this
        return "unknown"


def _paths(prefix: str | Path) -> tuple[Path, Path, Path]:
    """Return the ``(bin, idx, json)`` paths for a corpus prefix.

    Args:
        prefix: Path prefix, e.g. ``.../polyone/corpus``.

    Returns:
        The three component paths.
    """
    base = Path(prefix)
    return (
        base.with_name(base.name + ".bin"),
        base.with_name(base.name + ".idx"),
        base.with_name(base.name + ".json"),
    )


def dedup_sidecar_path(prefix: str | Path) -> Path:
    """Return the path of the optional deduplication-hash sidecar.

    Args:
        prefix: Corpus path prefix.

    Returns:
        ``<prefix>.dedup.u64``.
    """
    base = Path(prefix)
    return base.with_name(base.name + ".dedup.u64")


class TokenizedCorpusWriter:
    """Incremental single-writer builder for a :class:`TokenizedCorpus`.

    Sequences are appended in order and streamed to disk, so a 100M-sequence
    corpus never needs to be in RAM. The writer is the *only* process that may
    touch the files: the parallel builder in
    ``scripts/prepare_large_corpus.py`` fans conversion out to a worker pool but
    keeps writing in the parent, which is what keeps ``.bin`` ordered and the
    offsets consistent.

    Use it as a context manager -- :meth:`close` writes the metadata file, and a
    corpus without metadata is unreadable.

    Args:
        path_prefix: Path prefix for the three (or four) component files.
        tokenizer_sha256: SHA-256 of the vocabulary used to produce the ids.
        tokenizer_path: Where that vocabulary artifact lives (provenance).
        max_length: Token budget applied by the caller's length filter.
        dtype: Token-id dtype, ``"uint16"`` (default) or ``"uint32"``.
        vocab_size: Size of the vocabulary. Checked against the dtype up front
            so an oversized vocabulary is rejected before any bytes are
            written, rather than silently wrapping around.
        source_path: Raw input path, or list of them.
        append: Continue an existing corpus instead of truncating it.
        n_sequences: When appending, the sequence count to rewind to. The
            existing files are truncated to exactly this state first, which
            repairs a torn write from an interrupted job.
        n_tokens: When appending, the token count to rewind to.
        flush_tokens: Buffered ids before :meth:`add` writes through.

    Raises:
        ValueError: ``dtype`` is unsupported, or ``vocab_size`` does not fit it.
    """

    def __init__(
        self,
        path_prefix: str | Path,
        *,
        tokenizer_sha256: str,
        tokenizer_path: str | Path | None = None,
        max_length: int = 200,
        dtype: str = "uint16",
        vocab_size: int | None = None,
        source_path: str | Path | Sequence[str | Path] | None = None,
        append: bool = False,
        n_sequences: int | None = None,
        n_tokens: int | None = None,
        flush_tokens: int = _DEFAULT_FLUSH_TOKENS,
    ) -> None:
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
        self._dtype_name = dtype
        self._dtype = DTYPES[dtype]
        self._max_id = int(np.iinfo(self._dtype).max)
        if vocab_size is not None and int(vocab_size) > self._max_id + 1:
            raise ValueError(
                f"vocab_size {vocab_size} does not fit dtype {dtype!r} "
                f"(max id {self._max_id}); use dtype='uint32'"
            )

        self._prefix = Path(path_prefix)
        self._bin_path, self._idx_path, self._meta_path = _paths(self._prefix)
        self._bin_path.parent.mkdir(parents=True, exist_ok=True)

        self._tokenizer_sha256 = str(tokenizer_sha256)
        self._tokenizer_path = None if tokenizer_path is None else str(tokenizer_path)
        self._max_length = int(max_length)
        self._vocab_size = None if vocab_size is None else int(vocab_size)
        self._source_path = _normalize_source(source_path)
        self._flush_tokens = int(flush_tokens)
        self._preparation_stats: dict[str, int] = {}
        self._extra_metadata: dict[str, Any] = {}

        self._buf_ids: list[int] = []
        self._buf_lengths: list[int] = []
        self._closed = False

        if append and self._bin_path.exists() and self._idx_path.exists():
            self._n_sequences, self._n_tokens = self._rewind(n_sequences, n_tokens)
            self._bin_fh = self._bin_path.open("r+b")
            self._idx_fh = self._idx_path.open("r+b")
            self._bin_fh.seek(0, os.SEEK_END)
            self._idx_fh.seek(0, os.SEEK_END)
        else:
            self._n_sequences = 0
            self._n_tokens = 0
            self._bin_fh = self._bin_path.open("wb")
            self._idx_fh = self._idx_path.open("wb")
            # The leading 0 makes sequence 0 a plain slice like any other.
            np.zeros(1, dtype=OFFSET_DTYPE).tofile(self._idx_fh)

    # -- construction helpers ---------------------------------------------

    def _rewind(self, n_sequences: int | None, n_tokens: int | None) -> tuple[int, int]:
        """Truncate existing files to a known-good state before appending.

        Args:
            n_sequences: Target sequence count, or None to keep what is on disk.
            n_tokens: Target token count, or None to derive it from the offsets.

        Returns:
            ``(n_sequences, n_tokens)`` actually in the files afterwards.

        Raises:
            ValueError: The requested state is larger than what exists.
        """
        offsets = np.fromfile(self._idx_path, dtype=OFFSET_DTYPE)
        if offsets.size == 0:
            offsets = np.zeros(1, dtype=OFFSET_DTYPE)
            offsets.tofile(self._idx_path)
        have = int(offsets.size) - 1
        if n_sequences is None:
            n_sequences = have
        if n_sequences > have:
            raise ValueError(
                f"cannot append from sequence {n_sequences}: {self._idx_path} holds {have}"
            )
        derived = int(offsets[n_sequences])
        if n_tokens is not None and n_tokens != derived:
            raise ValueError(
                f"inconsistent resume point: idx says {derived} tokens at sequence "
                f"{n_sequences}, caller said {n_tokens}"
            )
        itemsize = int(np.dtype(self._dtype).itemsize)
        with self._idx_path.open("r+b") as fh:
            fh.truncate((n_sequences + 1) * int(np.dtype(OFFSET_DTYPE).itemsize))
        with self._bin_path.open("r+b") as fh:
            fh.truncate(derived * itemsize)
        return n_sequences, derived

    # -- metadata ----------------------------------------------------------

    def set_preparation_stats(self, stats: Any) -> None:
        """Attach attrition accounting to the corpus metadata.

        Args:
            stats: A :class:`polyt5.data.prepare.PreparationStats`, or any
                object with ``to_dict()``, or a plain mapping.
        """
        if hasattr(stats, "to_dict"):
            stats = stats.to_dict()
        self._preparation_stats = {str(k): int(v) for k, v in dict(stats).items()}

    def set_metadata(self, **fields: Any) -> None:
        """Merge extra JSON-serializable fields into the metadata.

        Args:
            **fields: Additional provenance to record (build settings, etc.).
        """
        self._extra_metadata.update(fields)

    # -- appending ---------------------------------------------------------

    def add(self, ids: Sequence[int]) -> None:
        """Append one sequence.

        Args:
            ids: Token ids of a single sequence; must be non-empty.

        Raises:
            ValueError: The sequence is empty, or an id does not fit the dtype.
            RuntimeError: The writer is closed.
        """
        self._check_open()
        n = len(ids)
        if n == 0:
            raise ValueError("refusing to write an empty sequence (it carries no loss signal)")
        self._buf_ids.extend(int(i) for i in ids)
        self._buf_lengths.append(n)
        if len(self._buf_ids) >= self._flush_tokens:
            self._flush_buffer()

    def add_many(self, sequences: Iterable[Sequence[int]]) -> None:
        """Append many sequences in order.

        Args:
            sequences: Iterable of token-id sequences.
        """
        for ids in sequences:
            self.add(ids)

    def add_packed(self, flat: np.ndarray, lengths: np.ndarray) -> None:
        """Append many sequences from an already-packed pair of arrays.

        This is the fast path used by ``scripts/prepare_large_corpus.py``: the
        workers return concatenated ids plus per-row lengths, so the parent
        never materializes a list of lists.

        Args:
            flat: 1-D array of concatenated ids; ``flat.size`` must equal
                ``lengths.sum()``.
            lengths: 1-D array of per-sequence lengths, all > 0.

        Raises:
            ValueError: The arrays disagree, a length is zero, or an id does
                not fit the dtype.
        """
        self._check_open()
        flat = np.asarray(flat)
        lengths = np.asarray(lengths, dtype=np.int64)
        if lengths.size == 0:
            return
        if int(lengths.sum()) != int(flat.size):
            raise ValueError(
                f"packed arrays disagree: lengths sum to {int(lengths.sum())}, "
                f"flat has {int(flat.size)} ids"
            )
        if int(lengths.min()) <= 0:
            raise ValueError("refusing to write an empty sequence (it carries no loss signal)")
        self._flush_buffer()
        self._write_arrays(flat, lengths)

    def _flush_buffer(self) -> None:
        """Write and clear the Python-side buffer used by :meth:`add`."""
        if not self._buf_lengths:
            return
        flat = np.asarray(self._buf_ids, dtype=np.int64)
        lengths = np.asarray(self._buf_lengths, dtype=np.int64)
        self._buf_ids = []
        self._buf_lengths = []
        self._write_arrays(flat, lengths)

    def _write_arrays(self, flat: np.ndarray, lengths: np.ndarray) -> None:
        """Validate and append packed ids plus their offsets.

        Args:
            flat: Concatenated ids.
            lengths: Per-sequence lengths.

        Raises:
            ValueError: An id is negative or exceeds the dtype's maximum.
        """
        if flat.size:
            lo = int(flat.min())
            hi = int(flat.max())
            if lo < 0 or hi > self._max_id:
                raise ValueError(
                    f"token id out of range for dtype {self._dtype_name!r}: "
                    f"found [{lo}, {hi}], allowed [0, {self._max_id}]"
                )
        flat.astype(self._dtype, copy=False).tofile(self._bin_fh)
        offsets = self._n_tokens + np.cumsum(lengths, dtype=np.int64)
        offsets.astype(OFFSET_DTYPE, copy=False).tofile(self._idx_fh)
        self._n_tokens = int(offsets[-1])
        self._n_sequences += int(lengths.size)

    # -- lifecycle ---------------------------------------------------------

    @property
    def n_sequences(self) -> int:
        """Sequences written so far (including buffered ones)."""
        return self._n_sequences + len(self._buf_lengths)

    @property
    def n_tokens(self) -> int:
        """Token ids written so far (including buffered ones)."""
        return self._n_tokens + len(self._buf_ids)

    def flush(self) -> tuple[int, int]:
        """Force every buffered sequence out to disk.

        Callers that checkpoint progress must flush *before* recording it, so a
        crash can only ever lose work, never invent it.

        Returns:
            ``(n_sequences, n_tokens)`` durably on disk.
        """
        self._check_open()
        self._flush_buffer()
        self._bin_fh.flush()
        self._idx_fh.flush()
        os.fsync(self._bin_fh.fileno())
        os.fsync(self._idx_fh.fileno())
        return self._n_sequences, self._n_tokens

    def write_metadata(self) -> dict[str, Any]:
        """Write ``<prefix>.json`` for the current state and return it.

        Returns:
            The metadata dict as written.
        """
        metadata: dict[str, Any] = {
            "format_version": CORPUS_FORMAT_VERSION,
            "n_sequences": self._n_sequences,
            "n_tokens": self._n_tokens,
            "dtype": self._dtype_name,
            "max_length": self._max_length,
            "vocab_size": self._vocab_size,
            "tokenizer_sha256": self._tokenizer_sha256,
            "tokenizer_path": self._tokenizer_path,
            "source_path": self._source_path,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "polyt5_version": _polyt5_version(),
            "preparation_stats": dict(self._preparation_stats),
        }
        metadata.update(self._extra_metadata)
        _atomic_write_json(self._meta_path, metadata)
        return metadata

    def close(self) -> dict[str, Any]:
        """Flush, write the metadata and close the file handles.

        Returns:
            The metadata dict as written.
        """
        if self._closed:
            return json.loads(self._meta_path.read_text(encoding="utf-8"))
        self.flush()
        metadata = self.write_metadata()
        self._bin_fh.close()
        self._idx_fh.close()
        self._closed = True
        return metadata

    def _check_open(self) -> None:
        """Raise if the writer has already been closed."""
        if self._closed:
            raise RuntimeError("TokenizedCorpusWriter is closed")

    def __enter__(self) -> TokenizedCorpusWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        elif not self._closed:
            # Keep whatever survived: the sidecar-based resume path can pick it
            # up, and half a corpus with honest metadata beats a corrupt one.
            try:
                self.flush()
                self.write_metadata()
            finally:
                self._bin_fh.close()
                self._idx_fh.close()
                self._closed = True

    def __repr__(self) -> str:
        return (
            f"TokenizedCorpusWriter({self._prefix}, n_sequences={self.n_sequences}, "
            f"n_tokens={self.n_tokens}, dtype={self._dtype_name})"
        )


class TokenizedCorpus:
    """Read-only, memory-mapped view of a corpus written by the writer.

    The maps are opened lazily and re-opened in every process that touches
    them (keyed by ``os.getpid()``), so the same instance can be handed to
    ``DataLoader(num_workers>0)`` under both fork and spawn.

    Args:
        path_prefix: Path prefix of the corpus files.
        metadata: Pre-loaded metadata; read from ``<prefix>.json`` when None.
    """

    def __init__(self, path_prefix: str | Path, metadata: dict[str, Any] | None = None) -> None:
        self._prefix = Path(path_prefix)
        self._bin_path, self._idx_path, self._meta_path = _paths(self._prefix)
        if metadata is None:
            if not self._meta_path.exists():
                raise FileNotFoundError(
                    f"no corpus metadata at {self._meta_path}; "
                    "was the corpus written by TokenizedCorpusWriter?"
                )
            metadata = json.loads(self._meta_path.read_text(encoding="utf-8"))
        self._metadata = metadata
        self._dtype_name = str(metadata.get("dtype", "uint16"))
        if self._dtype_name not in DTYPES:
            raise ValueError(f"corpus dtype {self._dtype_name!r} is not supported")
        self._dtype = DTYPES[self._dtype_name]
        self._n_sequences = int(metadata["n_sequences"])
        self._n_tokens = int(metadata["n_tokens"])
        self._tokens: np.memmap | np.ndarray | None = None
        self._offsets: np.memmap | np.ndarray | None = None
        self._pid: int | None = None

    @classmethod
    def from_prefix(cls, path_prefix: str | Path) -> TokenizedCorpus:
        """Open a corpus by path prefix.

        Args:
            path_prefix: Path prefix of the corpus files.

        Returns:
            The corpus. Nothing is mapped until the first access.
        """
        return cls(path_prefix)

    # -- process-safe lazy mapping ----------------------------------------

    def _arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(tokens, offsets)``, mapping them for this process first.

        Returns:
            Read-only arrays valid in the *calling* process.
        """
        pid = os.getpid()
        if self._tokens is None or self._offsets is None or self._pid != pid:
            # np.memmap refuses a zero-length file, and an empty corpus is a
            # legitimate (if useless) state -- e.g. every row failed the filters.
            if self._n_tokens == 0:
                self._tokens = np.empty(0, dtype=self._dtype)
            else:
                self._tokens = np.memmap(self._bin_path, dtype=self._dtype, mode="r")
            self._offsets = np.memmap(self._idx_path, dtype=OFFSET_DTYPE, mode="r")
            self._pid = pid
        return self._tokens, self._offsets

    def __getstate__(self) -> dict[str, Any]:
        """Drop the memory maps so the object pickles across a spawn."""
        state = dict(self.__dict__)
        state["_tokens"] = None
        state["_offsets"] = None
        state["_pid"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    # -- reading -----------------------------------------------------------

    def __len__(self) -> int:
        return self._n_sequences

    def __getitem__(self, index: int) -> list[int]:
        """Return sequence ``index`` as a plain list of ints.

        Args:
            index: Sequence index; negative indices count from the end.

        Returns:
            The token ids.

        Raises:
            IndexError: ``index`` is out of range.
        """
        i = int(index)
        if i < 0:
            i += self._n_sequences
        if not 0 <= i < self._n_sequences:
            raise IndexError(
                f"sequence index {index} out of range [0, {self._n_sequences})"
            )
        tokens, offsets = self._arrays()
        start = int(offsets[i])
        end = int(offsets[i + 1])
        return tokens[start:end].tolist()

    def __iter__(self) -> Iterator[list[int]]:
        for i in range(self._n_sequences):
            yield self[i]

    def lengths(self) -> np.ndarray:
        """Return the length of every sequence as an ``int64`` array.

        Returns:
            Array of shape ``(n_sequences,)``.
        """
        _, offsets = self._arrays()
        return np.diff(np.asarray(offsets, dtype=np.int64))

    def slice_view(self, indices: Sequence[int] | np.ndarray) -> TokenizedCorpusView:
        """Expose a subset (typically one split) without copying any tokens.

        Args:
            indices: Sequence indices, in the order the view should present.

        Returns:
            A :class:`TokenizedCorpusView` over the same memory map.
        """
        return TokenizedCorpusView(self, indices)

    # -- metadata ----------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        """The parsed ``<prefix>.json`` payload."""
        return self._metadata

    @property
    def n_tokens(self) -> int:
        """Total token ids stored in ``<prefix>.bin``."""
        return self._n_tokens

    @property
    def n_sequences(self) -> int:
        """Number of sequences."""
        return self._n_sequences

    @property
    def tokenizer_sha256(self) -> str:
        """SHA-256 of the vocabulary the corpus was tokenized with."""
        return str(self._metadata.get("tokenizer_sha256", ""))

    @property
    def dtype(self) -> str:
        """Name of the on-disk token dtype."""
        return self._dtype_name

    @property
    def prefix(self) -> Path:
        """The corpus path prefix."""
        return self._prefix

    def check_tokenizer(self, tokenizer: Any) -> None:
        """Assert this corpus was built with ``tokenizer``'s vocabulary.

        Args:
            tokenizer: Anything exposing ``sha256``.

        Raises:
            TokenizerMismatchError: The hashes differ.
        """
        expected = getattr(tokenizer, "sha256", None)
        if expected is None:
            raise ValueError("tokenizer has no sha256 attribute")
        if expected != self.tokenizer_sha256:
            raise TokenizerMismatchError(
                f"corpus {self._prefix} was tokenized with vocabulary "
                f"{self.tokenizer_sha256[:12]}... but the configured tokenizer is "
                f"{expected[:12]}...; the same id means a different token, so this is a "
                "different corpus. Rebuild it with scripts/prepare_large_corpus.py or "
                "point the config at the matching tokenizer artifact."
            )

    def __repr__(self) -> str:
        return (
            f"TokenizedCorpus({self._prefix}, n_sequences={self._n_sequences}, "
            f"n_tokens={self._n_tokens}, dtype={self._dtype_name})"
        )


class TokenizedCorpusView:
    """An index-selected view over a :class:`TokenizedCorpus`.

    Holds only the index array; the tokens stay in the parent's memory map, so
    a 90/5/5 split costs three small integer arrays rather than three corpora.

    Args:
        corpus: The underlying corpus.
        indices: Sequence indices, in presentation order.

    Raises:
        IndexError: Any index falls outside the corpus.
    """

    def __init__(self, corpus: TokenizedCorpus, indices: Sequence[int] | np.ndarray) -> None:
        self._corpus = corpus
        self._indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if self._indices.size:
            lo = int(self._indices.min())
            hi = int(self._indices.max())
            if lo < 0 or hi >= len(corpus):
                raise IndexError(
                    f"view indices [{lo}, {hi}] fall outside corpus of length {len(corpus)}"
                )

    def __len__(self) -> int:
        return int(self._indices.size)

    def __getitem__(self, index: int) -> list[int]:
        return self._corpus[int(self._indices[index])]

    def __iter__(self) -> Iterator[list[int]]:
        for i in range(len(self)):
            yield self[i]

    @property
    def indices(self) -> np.ndarray:
        """The underlying index array (int64)."""
        return self._indices

    @property
    def corpus(self) -> TokenizedCorpus:
        """The corpus this view reads from."""
        return self._corpus

    def __repr__(self) -> str:
        return f"TokenizedCorpusView(n={len(self)}, corpus={self._corpus.prefix})"


class MemmapPSelfiesDataset(torch.utils.data.Dataset):
    """Torch dataset over a memory-mapped corpus, optionally split-restricted.

    Each item is the raw ``list[int]`` of one sequence, exactly what
    :class:`polyt5.data.collate.SpanCorruptionCollator` expects; padding and
    span masking stay the collator's job.

    This is the drop-in replacement for
    :class:`polyt5.data.datasets.PSelfiesCorpus` at scale: no strings in RAM and
    no per-item tokenizer call, which is what closed the 19x epoch-0/epoch-1 gap.

    Args:
        corpus: A :class:`TokenizedCorpus` or :class:`TokenizedCorpusView`.
        indices: Optional split indices into ``corpus``; ignored (and rejected)
            when ``corpus`` is already a view.
        max_length: Optional truncation applied on read. The builder already
            enforces the token budget, so this is a belt-and-braces guard for
            corpora built with a larger budget than the model's ``n_positions``.

    Raises:
        ValueError: ``indices`` was passed together with a view.
    """

    def __init__(
        self,
        corpus: TokenizedCorpus | TokenizedCorpusView,
        indices: Sequence[int] | np.ndarray | None = None,
        *,
        max_length: int | None = None,
    ) -> None:
        if isinstance(corpus, TokenizedCorpusView):
            if indices is not None:
                raise ValueError("pass indices to slice_view(), not to a view-backed dataset")
            self._source: TokenizedCorpus | TokenizedCorpusView = corpus
        elif indices is None:
            self._source = corpus
        else:
            self._source = corpus.slice_view(indices)
        self._max_length = None if max_length is None else int(max_length)

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, index: int) -> list[int]:
        """Return one sequence's token ids."""
        ids = self._source[index]
        if self._max_length is not None and len(ids) > self._max_length:
            return ids[: self._max_length]
        return ids

    @property
    def corpus(self) -> TokenizedCorpus:
        """The underlying corpus."""
        source = self._source
        return source.corpus if isinstance(source, TokenizedCorpusView) else source

    @property
    def stats(self) -> dict[str, Any]:
        """Cheap descriptive statistics for run manifests."""
        corpus = self.corpus
        return {
            "n_examples": len(self),
            "n_corpus_sequences": len(corpus),
            "n_corpus_tokens": corpus.n_tokens,
            "max_length": self._max_length,
            "tokenizer_sha256": corpus.tokenizer_sha256,
            "prefix": str(corpus.prefix),
        }


def verify_corpus(
    prefix: str | Path,
    tokenizer: Any = None,
    *,
    sample_size: int = 4096,
) -> dict[str, Any]:
    """Sanity-check a corpus on disk and report what is in it.

    Checks that the component files are mutually consistent (offsets monotonic,
    file sizes matching the metadata), summarises sequence lengths, counts
    unknown tokens on an evenly spaced sample, and -- when a tokenizer is given
    -- asserts the vocabulary identity.

    Args:
        prefix: Corpus path prefix.
        tokenizer: Optional tokenizer exposing ``sha256`` and ``unk_id``.
        sample_size: Sequences to scan for unknown tokens; the sample is evenly
            spaced over the corpus, not the first N, so a corrupted tail shows up.

    Returns:
        A report dict with ``n_sequences``, ``n_tokens``, ``min_length``,
        ``max_length``, ``mean_length``, ``n_sampled``, ``n_unknown_tokens``,
        ``tokenizer_sha256``, ``tokenizer_sha256_matches`` and ``ok``.

    Raises:
        TokenizerMismatchError: The corpus was built with another vocabulary.
        ValueError: The files are inconsistent (truncated or torn write).
        FileNotFoundError: A component file is missing.
    """
    bin_path, idx_path, meta_path = _paths(prefix)
    for path in (bin_path, idx_path, meta_path):
        if not path.exists():
            raise FileNotFoundError(f"corpus component missing: {path}")

    corpus = TokenizedCorpus.from_prefix(prefix)
    if tokenizer is not None:
        corpus.check_tokenizer(tokenizer)

    itemsize = int(np.dtype(DTYPES[corpus.dtype]).itemsize)
    expected_bin = corpus.n_tokens * itemsize
    actual_bin = bin_path.stat().st_size
    if actual_bin != expected_bin:
        raise ValueError(
            f"{bin_path} is truncated or over-long: size {actual_bin} bytes, metadata "
            f"implies {expected_bin} ({corpus.n_tokens} x {itemsize})"
        )
    expected_idx = (len(corpus) + 1) * int(np.dtype(OFFSET_DTYPE).itemsize)
    actual_idx = idx_path.stat().st_size
    if actual_idx != expected_idx:
        raise ValueError(
            f"{idx_path} is truncated or over-long: size {actual_idx} bytes, metadata "
            f"implies {expected_idx} ({len(corpus)} + 1 offsets)"
        )

    lengths = corpus.lengths()
    if lengths.size and int(lengths.min()) <= 0:
        raise ValueError(f"{idx_path} has non-increasing offsets: empty or reversed sequence")

    n_unknown = 0
    n_sampled = 0
    if tokenizer is not None and len(corpus):
        unk_id = int(tokenizer.unk_id)
        step = max(1, len(corpus) // max(1, sample_size))
        for i in range(0, len(corpus), step):
            ids = corpus[i]
            n_sampled += 1
            n_unknown += sum(1 for t in ids if t == unk_id)

    matches = None if tokenizer is None else corpus.tokenizer_sha256 == tokenizer.sha256
    return {
        "prefix": str(Path(prefix)),
        "format_version": corpus.metadata.get("format_version"),
        "dtype": corpus.dtype,
        "n_sequences": len(corpus),
        "n_tokens": corpus.n_tokens,
        "min_length": int(lengths.min()) if lengths.size else 0,
        "max_length": int(lengths.max()) if lengths.size else 0,
        "mean_length": float(lengths.mean()) if lengths.size else 0.0,
        "max_length_setting": corpus.metadata.get("max_length"),
        "n_sampled": n_sampled,
        "n_unknown_tokens": n_unknown,
        "tokenizer_sha256": corpus.tokenizer_sha256,
        "tokenizer_sha256_matches": matches,
        "bytes_bin": actual_bin,
        "bytes_idx": actual_idx,
        "preparation_stats": corpus.metadata.get("preparation_stats", {}),
        "ok": True,
    }


def load_split_indices(splits_path: str | Path, name: str) -> np.ndarray:
    """Load one split's index array from a ``splits.json`` written by the builder.

    At 100M sequences an inline JSON index list is roughly 900 MB of text, which
    is unusable, so ``scripts/prepare_large_corpus.py`` writes large splits as
    ``.npy`` sidecars and stores ``{"npy": "<filename>"}`` in the JSON instead.
    This loader accepts both shapes, so callers never branch on corpus size.

    Args:
        splits_path: Path to ``splits.json``.
        name: Split name, e.g. ``"train"``.

    Returns:
        An ``int64`` array of corpus indices.

    Raises:
        KeyError: The split is absent from the file.
        FileNotFoundError: A referenced ``.npy`` sidecar is missing.
    """
    path = Path(splits_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if name not in payload:
        raise KeyError(f"split {name!r} not in {path} (have: {sorted(payload)})")
    entry = payload[name]
    if isinstance(entry, dict):
        sidecar = path.parent / str(entry["npy"])
        if not sidecar.exists():
            raise FileNotFoundError(f"split sidecar missing: {sidecar}")
        return np.load(sidecar).astype(np.int64, copy=False)
    return np.asarray(entry, dtype=np.int64)


def _normalize_source(
    source_path: str | Path | Sequence[str | Path] | None,
) -> str | list[str] | None:
    """Coerce a source-path argument into a JSON-friendly shape.

    Args:
        source_path: One path, several paths, or None.

    Returns:
        A string, a list of strings, or None.
    """
    if source_path is None:
        return None
    if isinstance(source_path, (str, Path)):
        return str(source_path)
    return [str(p) for p in source_path]


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON via a temporary file so a crash cannot leave half a file.

    Args:
        path: Destination path.
        payload: JSON-serializable object.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
