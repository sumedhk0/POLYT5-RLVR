"""Novelty at 100M-polymer scale: a sorted uint64 hash array, memory-mapped.

:class:`polyt5.chemistry.novelty.NoveltyIndex` keeps canonical PSMILES in a
Python ``set[str]``. That is the right structure for the ~7k fine-tuning corpus
and the wrong one for the pretraining corpus. This module is the scale variant,
used by the paper's TSD filter when the reference set is the *full* pretraining
corpus, and by the future RLVR novelty reward, which queries once per generated
candidate from CPU worker processes.

Why a sorted hash array
-----------------------

===========================  =========  =====================================
Structure (n = 1e8)          Footprint  Why not
===========================  =========  =====================================
``set[str]``                 10-15 GB   ~60 B of object overhead per string on
                                        top of the characters, private to each
                                        process, and rebuilt from scratch on
                                        every process start.
Bloom filter (fpr 1e-4)      ~240 MB    Smaller, but the false-positive rate is
                                        a *systematic* bias on the reported
                                        novelty rate -- one in 10^4 genuinely
                                        novel candidates is permanently and
                                        unfixably reported as "already seen" --
                                        and membership can never be confirmed.
Sorted ``uint64`` + memmap   800 MB     Chosen.
===========================  =========  =====================================

The arithmetic behind the choice:

* **Size.** 1e8 entries x 8 B = 800 MB on disk. Memory-mapped read-only, so the
  OS page cache holds *one* copy no matter how many rollout workers query it,
  and a worker that touches 1% of the pages resides in 8 MB, not 800 MB.
* **Speed.** Lookup is ``np.searchsorted``: log2(1e8) ~= 27 comparisons. A whole
  batch of candidates resolves in a single vectorized call, which is exactly the
  shape of the RL rollout loop's question ("of these 10k rollouts, which are
  new?"). The scalar path exists only for API compatibility.
* **Collisions.** With 64-bit hashes and n = 1e8 the expected number of
  colliding pairs is n^2 / 2^65 = 1e16 / 3.69e19 ~= **2.7e-4**. A false
  "already seen" verdict is therefore expected roughly once per 3700 corpora of
  this size -- five orders of magnitude below the corpus's own attrition noise,
  and unlike a Bloom filter's false-positive rate it does not scale with the
  number of queries.
* **Exactness.** Two indexes built from the same corpus are bit-identical
  files, diffable and checksummable. A Bloom filter's contents depend on its
  parameterisation and cannot be compared or enumerated.

Hash agreement is load bearing
------------------------------
The index is normally built from the ``<prefix>.dedup.u64`` sidecar that
``scripts/prepare_large_corpus.py`` writes as a by-product of corpus
preparation. :func:`hash64` here and ``_hash64`` there **must stay identical**;
a divergence does not raise, it silently reports every candidate as novel, which
looks like a spectacularly good model. ``tests/test_scalable_novelty.py`` pins
the two against each other and against hard-coded digests. If you change the
hash, change both and rebuild every index.

String spaces
-------------
The sidecar hashes whatever string the corpus builder deduplicated on: raw
PSMILES lines for ``--dedup exact``, RDKit-canonical PSMILES for
``--dedup canonical``. An index therefore lives in exactly one of
:data:`HASH_SPACES`, records it in its metadata, and refuses queries that
declare a different one -- hashing a PSMILES against a PSELFIES-built index
would report everything as novel, the most dangerous silent failure available
here.

Choices the paper does not make for us
--------------------------------------
# [AMBIGUITY] the paper does not say which corpus TSD's reference set is. It
describes TSD as "training set deduplication" without stating whether the
training set is the fine-tuning data or the full pretraining corpus. This module
exists to make the second reading affordable; which one is reported is the
caller's decision, recorded in the index metadata's ``source``.

# [AMBIGUITY] the paper does not say that TSD may be answered approximately.
Membership here is exact up to a 64-bit hash collision, whose expected count is
2.7e-4 over a 1e8-row corpus (above). A collision reports a genuinely novel
polymer as already-seen, i.e. it can only *understate* novelty, which is the
conservative direction for a claim about generating new materials.

# [AMBIGUITY] the paper does not say whether novelty is judged per generated
string or per distinct polymer. :meth:`ScalableNoveltyIndex.novelty_rate`
follows :func:`polyt5.chemistry.novelty.novelty_rate` and does not collapse
duplicates; callers wanting per-polymer novelty deduplicate first, as
``polyt5.evaluation.generation_metrics`` does.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "HASH_NAME",
    "HASH_SPACES",
    "INDEX_DTYPE",
    "HashSpaceMismatchError",
    "NoveltyIndexMetadata",
    "ScalableNoveltyIndex",
    "hash64",
    "hash64_for_space",
    "hash64_many",
]

#: The three string spaces an index may be built in. ``canonical_psmiles`` is
#: RDKit-canonical PSMILES (what ``--dedup canonical`` hashes), ``psmiles`` is
#: the raw line as written (what ``--dedup exact`` hashes), ``pselfies`` is the
#: encoded form.
HASH_SPACES: tuple[str, ...] = ("pselfies", "canonical_psmiles", "psmiles")

#: Name of the hash recorded in the metadata. Any change to :func:`hash64`
#: must change this string, so an old index cannot be silently reused.
HASH_NAME = "blake2b-64"

#: On-disk element type. Native byte order, matching the ``.dedup.u64``
#: sidecar, which is written with ``ndarray.tobytes()``.
INDEX_DTYPE = np.uint64

#: ``scripts/prepare_large_corpus.py --dedup <mode>`` to hash space.
_DEDUP_MODE_TO_SPACE: dict[str, str] = {
    "exact": "psmiles",
    "canonical": "canonical_psmiles",
}

#: Spaces whose members are PSMILES strings, and which may therefore answer the
#: ``NoveltyIndex``-compatible surface (``x in index`` / ``is_novel(x)``).
_PSMILES_SPACES = frozenset({"psmiles", "canonical_psmiles"})


class HashSpaceMismatchError(ValueError):
    """Raised when a query is asked in a different string space than the index.

    This is the one deliberate exception in this module. Answering such a query
    would report every candidate as novel, so it is refused instead.
    """


def hash64(text: str) -> int:
    """Hash a string to a 64-bit int, identically to the corpus builder.

    Must stay byte-for-byte identical to ``scripts/prepare_large_corpus.py``'s
    ``_hash64``: blake2b with an 8-byte digest, read big-endian. See the module
    docstring for why.

    Args:
        text: String to hash. Non-strings are coerced with ``str``.

    Returns:
        A 64-bit unsigned integer.
    """
    if not isinstance(text, str):
        text = str(text)
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def hash64_many(texts: Iterable[str]) -> np.ndarray:
    """Hash an iterable of strings to a ``uint64`` array.

    Args:
        texts: Strings to hash, in order.

    Returns:
        A ``uint64`` array of the same length, in input order.
    """
    return np.fromiter(
        (hash64(text) for text in texts), dtype=INDEX_DTYPE, count=-1
    )


def _resolve_canonicalize(hash_space: str, canonicalize: bool | None) -> bool:
    """Decide whether strings must be canonicalized before hashing.

    Canonicalization is a property of the hash space, not a free choice: an
    index that canonicalizes on build but not on query (or the reverse) answers
    nonsense. The flag exists so a caller can state the intent explicitly, and
    contradicting the space is a programming error.

    Args:
        hash_space: One of :data:`HASH_SPACES`.
        canonicalize: Caller's request, or ``None`` for "whatever the space
            implies".

    Returns:
        ``True`` if strings must be passed through
        :func:`polyt5.chemistry.canonicalization.canonical_psmiles` first.

    Raises:
        ValueError: If the request contradicts the hash space.
    """
    implied = hash_space == "canonical_psmiles"
    if canonicalize is None:
        return implied
    if bool(canonicalize) != implied:
        raise ValueError(
            f"canonicalize={canonicalize!r} contradicts hash_space={hash_space!r}: "
            "'canonical_psmiles' always canonicalizes, the other spaces never do"
        )
    return implied


def _validate_hash_space(hash_space: str) -> str:
    """Check a hash-space name.

    Args:
        hash_space: Candidate name.

    Returns:
        The name, unchanged.

    Raises:
        ValueError: If it is not one of :data:`HASH_SPACES`.
    """
    if hash_space not in HASH_SPACES:
        raise ValueError(f"hash_space must be one of {HASH_SPACES!r}, got {hash_space!r}")
    return hash_space


def _encode(texts: Iterable[Any], canonicalize: bool) -> tuple[np.ndarray, np.ndarray]:
    """Hash strings for indexing or querying, flagging the unusable ones.

    Total by construction: a value that is not a usable string in this space
    (unparseable PSMILES, un-encodable surrogates, ``None``) is flagged invalid
    rather than raising, because model output is adversarial by nature.

    Leading and trailing whitespace is stripped first, matching
    ``scripts/prepare_large_corpus.py``, which hashes ``raw.strip()``.

    Args:
        texts: Values to hash.
        canonicalize: Canonicalize each string before hashing.

    Returns:
        A ``(codes, valid)`` pair of equal-length arrays: ``uint64`` hashes,
        and a bool mask that is ``False`` where the value could not be hashed.
    """
    to_canonical = None
    if canonicalize:
        # Imported lazily: the exact spaces never need rdkit.
        from .canonicalization import canonical_psmiles

        to_canonical = canonical_psmiles

    codes: list[int] = []
    valid: list[bool] = []
    for text in texts:
        if not isinstance(text, str):
            codes.append(0)
            valid.append(False)
            continue
        prepared: str | None = text.strip()
        if to_canonical is not None:
            prepared = to_canonical(prepared) if prepared else None
        if not prepared:
            codes.append(0)
            valid.append(False)
            continue
        try:
            codes.append(hash64(prepared))
        except Exception:  # pragma: no cover - lone surrogates only
            codes.append(0)
            valid.append(False)
            continue
        valid.append(True)
    return (
        np.array(codes, dtype=INDEX_DTYPE),
        np.array(valid, dtype=bool),
    )


def hash64_for_space(
    texts: Iterable[str], *, hash_space: str, canonicalize: bool | None = None
) -> np.ndarray:
    """Hash strings the way an index in ``hash_space`` stores them.

    Unhashable entries are dropped, matching
    :class:`polyt5.chemistry.novelty.NoveltyIndex`, which skips reference rows
    it cannot canonicalize rather than storing unmatchable ghosts. Streaming
    builders (``scripts/build_novelty_index.py``) call this per chunk so the
    strings never accumulate in RAM.

    Args:
        texts: Strings to hash.
        hash_space: One of :data:`HASH_SPACES`.
        canonicalize: See :func:`_resolve_canonicalize`.

    Returns:
        A ``uint64`` array of the hashes that could be computed, unsorted and
        possibly shorter than the input.
    """
    space = _validate_hash_space(hash_space)
    do_canonical = _resolve_canonicalize(space, canonicalize)
    codes, valid = _encode(texts, do_canonical)
    return codes[valid]


def _sort_unique(hashes: np.ndarray) -> np.ndarray:
    """Return a sorted, deduplicated ``uint64`` copy of ``hashes``.

    Sorts in place on a private copy and dedups with a neighbour comparison
    rather than :func:`numpy.unique`, whose argsort path costs an extra 8 bytes
    per entry -- 800 MB at the scale this module exists for.

    Args:
        hashes: Any integer array.

    Returns:
        A strictly increasing ``uint64`` array.
    """
    array = np.ascontiguousarray(hashes, dtype=INDEX_DTYPE).copy()
    if array.size == 0:
        return array
    array.sort()
    keep = np.empty(array.size, dtype=bool)
    keep[0] = True
    np.not_equal(array[1:], array[:-1], out=keep[1:])
    if keep.all():
        return array
    return array[keep]


def _index_paths(path: str | Path) -> tuple[Path, Path]:
    """Return the ``(data, metadata)`` paths for an index location.

    Accepts the stem (``.../train_index``) or either component file, so a
    caller can pass back whatever it was handed.

    Args:
        path: Index location.

    Returns:
        The ``<path>.u64`` and ``<path>.json`` paths.
    """
    base = Path(path)
    if base.suffix in (".u64", ".json"):
        base = base.with_suffix("")
    return (
        base.with_name(base.name + ".u64"),
        base.with_name(base.name + ".json"),
    )


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class NoveltyIndexMetadata:
    """What an index is, so it cannot be misused.

    Attributes:
        n_entries: Number of distinct hashes stored.
        hash_space: The string space the hashes were taken in; one of
            :data:`HASH_SPACES`. Queries in another space are refused.
        hash_name: Identifier of the hash function, ``"blake2b-64"``. A change
            to :func:`hash64` must change this so old indexes are rejected.
        source: Human-readable provenance -- corpus prefix, file paths.
        created_utc: ISO-8601 build time.
        dtype: On-disk element type, always ``"uint64"``.
    """

    n_entries: int
    hash_space: str
    hash_name: str = HASH_NAME
    source: str = ""
    created_utc: str = field(default_factory=_utc_now)
    dtype: str = "uint64"

    def __post_init__(self) -> None:
        """Validate the declared space and dtype.

        Raises:
            ValueError: If ``hash_space`` or ``dtype`` is unknown.
        """
        _validate_hash_space(self.hash_space)
        if self.dtype != "uint64":
            raise ValueError(f"dtype must be 'uint64', got {self.dtype!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of the metadata."""
        return {
            "n_entries": int(self.n_entries),
            "hash_space": self.hash_space,
            "hash_name": self.hash_name,
            "source": self.source,
            "created_utc": self.created_utc,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NoveltyIndexMetadata:
        """Rebuild metadata from its JSON form.

        Args:
            payload: A dict as produced by :meth:`to_dict`.

        Returns:
            The metadata object.

        Raises:
            KeyError: If a required field is missing.
        """
        return cls(
            n_entries=int(payload["n_entries"]),
            hash_space=str(payload["hash_space"]),
            hash_name=str(payload.get("hash_name", HASH_NAME)),
            source=str(payload.get("source", "")),
            created_utc=str(payload.get("created_utc", "")),
            dtype=str(payload.get("dtype", "uint64")),
        )


class ScalableNoveltyIndex:
    """Membership over up to ~1e9 polymers, backed by sorted 64-bit hashes.

    Two backings, one interface. :meth:`from_hashes`, :meth:`from_strings` and
    :meth:`from_dedup_sidecar` hold the array in RAM (the builder's view);
    :meth:`open` memory-maps an already-built file read-only and never
    materialises it (the reward worker's view). The memmap is reopened whenever
    the process id changes, so an index may be pickled to a ``multiprocessing``
    worker -- Windows spawns rather than forks, and a memmap object does not
    survive that.

    Query surface, and which half of it carries a PSMILES contract:

    * ``x in index``, :meth:`is_novel` -- the surface shared with
      :class:`polyt5.chemistry.novelty.NoveltyIndex`, so ``x`` is a PSMILES.
      A PSELFIES-space index refuses these outright rather than reporting
      everything novel.
    * :meth:`contains_many`, :meth:`novelty_mask`, :meth:`novelty_rate` -- the
      vectorized surface and the hot path. These answer in the index's own
      space by default.
    * Every method takes an optional ``hash_space``; declaring one that differs
      from the index's raises :class:`HashSpaceMismatchError`.
    """

    def __init__(
        self,
        hashes: np.ndarray | None,
        metadata: NoveltyIndexMetadata,
        *,
        data_path: Path | None = None,
    ) -> None:
        """Construct from an in-memory array or a memmappable file.

        Prefer the classmethods; this signature is the plumbing they share.

        Args:
            hashes: A sorted, deduplicated ``uint64`` array, or ``None`` when
                backed by a file.
            metadata: The index's metadata. Its ``n_entries`` is authoritative
                for file-backed indexes and is checked on open.
            data_path: Path of the ``.u64`` file, when file-backed.

        Raises:
            ValueError: If neither or both backings are supplied.
        """
        if (hashes is None) == (data_path is None):
            raise ValueError("supply exactly one of hashes or data_path")
        self._hashes = None if hashes is None else np.ascontiguousarray(hashes, INDEX_DTYPE)
        self._data_path = None if data_path is None else Path(data_path)
        self._metadata = metadata
        self._mm: np.ndarray | None = None
        self._mm_pid: int | None = None

    # -- construction ------------------------------------------------------

    @classmethod
    def from_hashes(
        cls, hashes: np.ndarray, *, metadata: NoveltyIndexMetadata
    ) -> ScalableNoveltyIndex:
        """Build an in-memory index from raw hashes.

        Args:
            hashes: ``uint64`` hashes in any order, duplicates allowed.
            metadata: Metadata for the index. ``n_entries`` is recomputed from
                the deduplicated array, so callers may pass ``0``.

        Returns:
            The index.
        """
        sorted_hashes = _sort_unique(hashes)
        resolved = NoveltyIndexMetadata(
            n_entries=int(sorted_hashes.size),
            hash_space=metadata.hash_space,
            hash_name=metadata.hash_name,
            source=metadata.source,
            created_utc=metadata.created_utc,
            dtype=metadata.dtype,
        )
        return cls(sorted_hashes, resolved)

    @classmethod
    def from_strings(
        cls,
        strings: Iterable[str],
        *,
        hash_space: str,
        canonicalize: bool | None = None,
        source: str = "",
    ) -> ScalableNoveltyIndex:
        """Build an index by hashing strings.

        Entries that cannot be hashed in this space (unparseable PSMILES, when
        the space canonicalizes) are skipped, exactly as
        :class:`polyt5.chemistry.novelty.NoveltyIndex` skips them.

        Args:
            strings: Reference strings.
            hash_space: One of :data:`HASH_SPACES`.
            canonicalize: Canonicalize before hashing. ``None`` means "as the
                space implies"; contradicting the space raises. Canonicalizing
                costs an RDKit parse plus a canonical-SMILES write per string,
                roughly 10-40 us, i.e. hours for a 100M-row corpus -- which is
                why the corpus builder does it once and this module reads its
                sidecar instead.
            source: Provenance string for the metadata.

        Returns:
            The index.
        """
        space = _validate_hash_space(hash_space)
        do_canonical = _resolve_canonicalize(space, canonicalize)
        codes, valid = _encode(strings, do_canonical)
        metadata = NoveltyIndexMetadata(
            n_entries=0, hash_space=space, source=source
        )
        return cls.from_hashes(codes[valid], metadata=metadata)

    @classmethod
    def from_dedup_sidecar(
        cls, corpus_prefix: str | Path, *, hash_space: str | None = None
    ) -> ScalableNoveltyIndex:
        """Build from the ``<prefix>.dedup.u64`` written by corpus preparation.

        This is the cheap path and the intended one: the hashes already exist
        as a by-product of ``scripts/prepare_large_corpus.py``, so a 100M-row
        index is an 800 MB read and a sort, not a re-canonicalization of the
        whole corpus.

        Args:
            corpus_prefix: Corpus path prefix, e.g. ``.../polyone/corpus``.
            hash_space: Assert the space the corpus was deduplicated in. When
                ``None`` it is read from the corpus metadata's ``dedup`` field.

        Returns:
            The index, in the corpus's own hash space.

        Raises:
            FileNotFoundError: If the sidecar does not exist.
            ValueError: If the corpus was built with ``--dedup none``, or the
                space cannot be determined and was not asserted.
            HashSpaceMismatchError: If the asserted space is not the corpus's.
        """
        prefix = Path(corpus_prefix)
        sidecar = _dedup_sidecar_path(prefix)
        if not sidecar.exists():
            raise FileNotFoundError(
                f"no deduplication sidecar at {sidecar}; the corpus must be prepared "
                "with --dedup exact or --dedup canonical"
            )

        corpus_space: str | None = None
        meta_path = prefix.with_name(prefix.name + ".json")
        if meta_path.exists():
            corpus_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            dedup_mode = str(corpus_metadata.get("dedup", ""))
            if dedup_mode == "none":
                raise ValueError(
                    f"corpus {prefix} was prepared with --dedup none; its sidecar, if any, "
                    "is stale and must not be used as a novelty index"
                )
            corpus_space = _DEDUP_MODE_TO_SPACE.get(dedup_mode)

        if corpus_space is None:
            if hash_space is None:
                raise ValueError(
                    f"cannot determine the hash space of {sidecar}: no readable corpus "
                    f"metadata at {meta_path}. Pass hash_space explicitly, and be sure: a "
                    "wrong space reports every candidate as novel."
                )
            corpus_space = _validate_hash_space(hash_space)
        elif hash_space is not None and _validate_hash_space(hash_space) != corpus_space:
            raise HashSpaceMismatchError(
                f"corpus {prefix} was deduplicated in {corpus_space!r}, not {hash_space!r}; "
                "an index built on the wrong space reports every candidate as novel"
            )

        hashes: np.ndarray = np.fromfile(sidecar, dtype=INDEX_DTYPE)
        metadata = NoveltyIndexMetadata(
            n_entries=0, hash_space=corpus_space, source=str(prefix)
        )
        return cls.from_hashes(hashes, metadata=metadata)

    @classmethod
    def open(cls, path: str | Path) -> ScalableNoveltyIndex:
        """Memory-map an index written by :meth:`save`, read-only.

        Nothing is read here beyond the metadata; the array is mapped lazily on
        first query and remapped after a fork or spawn.

        Args:
            path: The index stem, or either of its component files.

        Returns:
            A file-backed index.

        Raises:
            FileNotFoundError: If either component file is missing.
            ValueError: If the file size is not a whole number of entries or
                disagrees with the metadata.
        """
        data_path, meta_path = _index_paths(path)
        if not data_path.exists():
            raise FileNotFoundError(f"no novelty index array at {data_path}")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no novelty index metadata at {meta_path}; an index without its hash "
                "space is unusable"
            )
        metadata = NoveltyIndexMetadata.from_dict(
            json.loads(meta_path.read_text(encoding="utf-8"))
        )
        size = data_path.stat().st_size
        itemsize = np.dtype(INDEX_DTYPE).itemsize
        if size % itemsize:
            raise ValueError(f"{data_path} is {size} bytes, not a multiple of {itemsize}")
        n_on_disk = size // itemsize
        if n_on_disk != metadata.n_entries:
            raise ValueError(
                f"{data_path} holds {n_on_disk} entries but {meta_path} declares "
                f"{metadata.n_entries}; the index is truncated or mismatched"
            )
        return cls(None, metadata, data_path=data_path)

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write ``<path>.u64`` and ``<path>.json``.

        Both files are written to a temporary name and renamed, so a crash
        mid-write cannot leave a half-index that opens successfully.

        Args:
            path: Destination stem. Parent directories are created.

        Returns:
            The path of the written ``.u64`` array.
        """
        data_path, meta_path = _index_paths(path)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        array = self._array()

        data_tmp = data_path.with_name(data_path.name + ".tmp")
        with data_tmp.open("wb") as handle:
            np.ascontiguousarray(array, dtype=INDEX_DTYPE).tofile(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(data_tmp, data_path)

        meta_tmp = meta_path.with_name(meta_path.name + ".tmp")
        meta_tmp.write_text(
            json.dumps(self._metadata.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        os.replace(meta_tmp, meta_path)
        return data_path

    # -- backing array -----------------------------------------------------

    def _array(self) -> np.ndarray:
        """Return the sorted hash array, mapping the file if necessary.

        The memmap is keyed on the process id: a spawned or forked child gets
        its own mapping instead of inheriting a handle that is not valid there.

        Returns:
            A sorted, strictly increasing ``uint64`` array.
        """
        if self._hashes is not None:
            return self._hashes
        assert self._data_path is not None  # guaranteed by __init__
        if self._metadata.n_entries == 0:
            # np.memmap refuses a zero-length file; an empty index is legal.
            return np.empty(0, dtype=INDEX_DTYPE)
        pid = os.getpid()
        if self._mm is None or self._mm_pid != pid:
            self._mm = np.memmap(self._data_path, dtype=INDEX_DTYPE, mode="r")
            self._mm_pid = pid
        return self._mm

    def __getstate__(self) -> dict[str, Any]:
        """Drop the memmap when pickling, so the child remaps it itself."""
        state = self.__dict__.copy()
        state["_mm"] = None
        state["_mm_pid"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from a pickle without touching the file."""
        self.__dict__.update(state)

    # -- hash-space discipline --------------------------------------------

    @property
    def metadata(self) -> NoveltyIndexMetadata:
        """The index's metadata."""
        return self._metadata

    @property
    def hash_space(self) -> str:
        """The string space this index answers questions in."""
        return self._metadata.hash_space

    @property
    def hashes(self) -> np.ndarray:
        """The sorted, deduplicated hash array.

        For a file-backed index this is the memmap itself, so touching it reads
        pages; slice it rather than copying 800 MB.
        """
        return self._array()

    def _resolve_query_space(self, hash_space: str | None, *, psmiles_contract: bool) -> str:
        """Check the space a query is asked in.

        Args:
            hash_space: The caller's declared space, or ``None`` for "the
                index's own".
            psmiles_contract: True for the ``NoveltyIndex``-compatible surface,
                whose argument is a PSMILES by contract. A PSELFIES-space index
                cannot honour that contract and refuses.

        Returns:
            The space to hash in.

        Raises:
            HashSpaceMismatchError: If the declared space is not the index's,
                or the PSMILES contract cannot be met.
        """
        if hash_space is not None:
            declared = _validate_hash_space(hash_space)
            if declared != self._metadata.hash_space:
                raise HashSpaceMismatchError(
                    f"this index was built in {self._metadata.hash_space!r} but the query "
                    f"declares {declared!r}; answering would report every candidate as novel"
                )
            return declared
        if psmiles_contract and self._metadata.hash_space not in _PSMILES_SPACES:
            raise HashSpaceMismatchError(
                f"'x in index' and is_novel() take a PSMILES, but this index was built in "
                f"{self._metadata.hash_space!r}; use contains_many(texts, "
                f"hash_space={self._metadata.hash_space!r}) to query it in its own space"
            )
        return self._metadata.hash_space

    # -- queries -----------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of distinct hashes in the index."""
        return int(self._metadata.n_entries)

    def contains_many(
        self,
        texts: Iterable[str],
        *,
        hash_space: str | None = None,
        already_canonical: bool = False,
    ) -> np.ndarray:
        """Test membership for a whole batch, in one vectorized search.

        This is the hot path: an RL rollout asks "which of these 10k candidates
        are already known?" and gets one ``np.searchsorted`` over the memmap
        rather than 10k Python-level lookups.

        Args:
            texts: Candidate strings, in the index's space.
            hash_space: Optionally assert the space; see
                :class:`HashSpaceMismatchError`.
            already_canonical: Skip canonicalization because the caller has
                already done it. Measured on this repo's hardware, hashing and
                searching costs ~0.6 us per candidate while canonicalizing costs
                ~310 us, so a caller that already holds the canonical PSMILES --
                :func:`polyt5.evaluation.apply_filter_cascade` computes one at
                the SV stage, and the RLVR reward will too -- should say so.
                Ignored for spaces that do not canonicalize. If the caller is
                wrong and the string is *not* canonical, its hash will match
                nothing and the candidate is reported novel.

        Returns:
            A bool array, ``True`` where the candidate is already in the index.
            Entries that cannot be hashed are ``False`` -- they match nothing.

        Raises:
            HashSpaceMismatchError: If ``hash_space`` is not the index's.
        """
        space = self._resolve_query_space(hash_space, psmiles_contract=False)
        return self._contains_batch(texts, space, already_canonical=already_canonical)

    def _contains_batch(
        self, texts: Iterable[str], space: str, *, already_canonical: bool = False
    ) -> np.ndarray:
        """Membership for a batch whose space has already been checked.

        Args:
            texts: Candidate strings.
            space: The resolved hash space.
            already_canonical: Trust the caller and skip canonicalization.

        Returns:
            A bool array, ``True`` where the candidate is in the index.
        """
        canonicalize = _resolve_canonicalize(space, None) and not already_canonical
        codes, valid = _encode(texts, canonicalize)
        if codes.size == 0:
            return np.zeros(0, dtype=bool)
        array = self._array()
        if array.size == 0:
            return np.zeros(codes.size, dtype=bool)
        positions = np.searchsorted(array, codes)
        np.minimum(positions, array.size - 1, out=positions)
        found = np.asarray(array[positions] == codes)
        found &= valid
        return found

    def novelty_mask(
        self,
        texts: Iterable[str],
        *,
        hash_space: str | None = None,
        already_canonical: bool = False,
    ) -> np.ndarray:
        """Return ``True`` where a candidate is *absent* from the index.

        Args:
            texts: Candidate strings, in the index's space.
            hash_space: Optionally assert the space.
            already_canonical: See :meth:`contains_many`.

        Returns:
            A bool array, the complement of :meth:`contains_many`.

        Note:
            A candidate that cannot be hashed matches nothing and is therefore
            reported novel. Novelty is meaningless for invalid structures, so
            callers must gate on validity first --
            :func:`polyt5.evaluation.apply_filter_cascade` does exactly that by
            running SV before TSD.
        """
        return ~self.contains_many(
            texts, hash_space=hash_space, already_canonical=already_canonical
        )

    def novelty_rate(
        self,
        texts: Iterable[str],
        *,
        hash_space: str | None = None,
        already_canonical: bool = False,
    ) -> float:
        """Fraction of candidates absent from the index.

        Args:
            texts: Candidate strings. Duplicates are *not* collapsed; pass a
                deduplicated sequence for novelty per distinct polymer.
            hash_space: Optionally assert the space.
            already_canonical: See :meth:`contains_many`.

        Returns:
            Novel candidates divided by total, or ``0.0`` for an empty input --
            matching :func:`polyt5.chemistry.novelty.novelty_rate`.
        """
        mask = self.novelty_mask(
            texts, hash_space=hash_space, already_canonical=already_canonical
        )
        if mask.size == 0:
            return 0.0
        return float(mask.mean())

    def self_check(
        self,
        known_members: Sequence[str],
        *,
        hash_space: str | None = None,
        already_canonical: bool = False,
        min_hit_rate: float = 1.0,
    ) -> float:
        """Assert that strings known to be in the index are actually found.

        The declared-space guard in :meth:`_resolve_query_space` only fires when
        a caller *declares* the wrong space. Nothing in Python distinguishes a
        PSELFIES string from a PSMILES string by type, so an undeclared query in
        the wrong space is silently answered "novel" for every candidate — the
        most dangerous failure this class has, because it looks like a great
        novelty result.

        This method catches that empirically: feed it a sample drawn from the
        very corpus the index was built from, and it must find them.

        Args:
            known_members: Strings that are definitely in the index, expressed
                in the index's own hash space.
            hash_space: Declared query space; see :meth:`contains_many`.
            already_canonical: Skip canonicalization for a canonical index.
            min_hit_rate: Minimum acceptable fraction found. The default of 1.0
                is correct when the sample really was drawn from the index;
                lower it only for a deliberately noisy sample.

        Returns:
            The measured hit rate.

        Raises:
            HashSpaceMismatchError: If the hit rate falls below ``min_hit_rate``,
                which in practice means the index and the query disagree about
                what is being hashed.
        """
        if not known_members:
            return 1.0
        found = ~self.novelty_mask(
            known_members, hash_space=hash_space, already_canonical=already_canonical
        )
        hit_rate = float(found.mean())
        if hit_rate < min_hit_rate:
            raise HashSpaceMismatchError(
                f"self-check failed: only {hit_rate:.1%} of {len(known_members)} known members "
                f"were found in an index built in {self._metadata.hash_space!r} "
                f"(source {self._metadata.source!r}). The strings you are querying with are "
                "almost certainly in a different space than the index was built in — for "
                "example querying PSELFIES against a PSMILES-space index, or non-canonical "
                "text against a canonical index."
            )
        return hit_rate

    def __contains__(self, psmiles: str) -> bool:
        """Test whether one polymer is already known.

        Args:
            psmiles: Candidate PSMILES. For a ``canonical_psmiles`` index
                either terminus notation works, as the string is canonicalized
                first; for a ``psmiles`` index the match is on the exact text,
                as written by the corpus builder.

        Returns:
            ``True`` if the candidate is in the index.

        Raises:
            HashSpaceMismatchError: If the index is not in a PSMILES space.
        """
        space = self._resolve_query_space(None, psmiles_contract=True)
        return bool(self._contains_batch([psmiles], space)[0])

    def is_novel(self, psmiles: str, *, hash_space: str | None = None) -> bool:
        """Test whether a candidate is absent from the reference set.

        Args:
            psmiles: Candidate string, a PSMILES unless ``hash_space`` says
                otherwise.
            hash_space: Optionally assert the space. Required to query a
                ``pselfies`` index through this method.

        Returns:
            ``True`` if the candidate is not in the index.

        Raises:
            HashSpaceMismatchError: If the declared space is not the index's,
                or no space was declared for a non-PSMILES index.
        """
        space = self._resolve_query_space(hash_space, psmiles_contract=hash_space is None)
        return not bool(self._contains_batch([psmiles], space)[0])

    def __repr__(self) -> str:
        backing = "memmap" if self._hashes is None else "memory"
        return (
            f"{type(self).__name__}(n={len(self)}, space={self._metadata.hash_space!r}, "
            f"backing={backing})"
        )


def _dedup_sidecar_path(prefix: Path) -> Path:
    """Return ``<prefix>.dedup.u64``, deferring to the corpus module if it loads.

    Imported lazily: ``polyt5.data`` is torch-free but not free, and a reward
    worker that only queries an index should not pay for the data pipeline.

    Args:
        prefix: Corpus path prefix.

    Returns:
        The sidecar path.
    """
    try:
        from polyt5.data.tokenized_corpus import dedup_sidecar_path
    except Exception:  # pragma: no cover - only if polyt5.data is unavailable
        return prefix.with_name(prefix.name + ".dedup.u64")
    return Path(dedup_sidecar_path(prefix))
