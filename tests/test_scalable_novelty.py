"""Tests for the memory-mapped, hash-based novelty index.

The single most important test in this file is
:func:`test_hash64_matches_prepare_large_corpus`. The scalable index is built
from the ``.dedup.u64`` sidecar that ``scripts/prepare_large_corpus.py`` writes,
so if the two hash functions ever diverge the index silently reports *every*
candidate as novel -- a wrong result that looks like a spectacularly good model.
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from polyt5.chemistry.conversion import pselfies_to_psmiles
from polyt5.chemistry.novelty import NoveltyIndex, novelty_rate
from polyt5.chemistry.scalable_novelty import (
    HASH_NAME,
    HASH_SPACES,
    HashSpaceMismatchError,
    NoveltyIndexMetadata,
    ScalableNoveltyIndex,
    hash64,
    hash64_many,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = REPO_ROOT / "scripts" / "prepare_large_corpus.py"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_novelty_index.py"

#: A real corpus prepared by another track; present on the dev box only.
POLYONE_DEV_PREFIX = Path("C:/Users/sumedh/polyt5-data/processed/polyone_dev/corpus")

#: A handful of small polymers used across the fast tests.
TOY_PSMILES = [
    "[*]CC[*]",
    "[*]CCO[*]",
    "[*]CC(C)[*]",
    "[*]c1ccccc1[*]",
    "[*]CC(=O)O[*]",
]
UNRELATED_PSMILES = [
    "[*]CCCCCCCCCCCCCCCCCCN[*]",
    "[*]C(F)(F)C(Cl)(Br)[*]",
    "[*]OCCOCCOCCOCCN(C)[*]",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _load_prepare_large_corpus():
    """Import ``scripts/prepare_large_corpus.py`` as a module, by path.

    Returns:
        The loaded module, or ``None`` if it cannot be imported.
    """
    spec = importlib.util.spec_from_file_location("_plc_under_test", PREPARE_SCRIPT)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:  # pragma: no cover - only if the sibling script breaks
        return None
    return module


def _real_source_lines(limit: int = 300) -> list[str]:
    """Read raw PSMILES lines from the corpus the dev sidecar was built from.

    Args:
        limit: Maximum number of lines to return.

    Returns:
        Stripped, non-empty lines, or an empty list when the file is absent.
    """
    meta_path = POLYONE_DEV_PREFIX.with_name(POLYONE_DEV_PREFIX.name + ".json")
    if not meta_path.exists():
        return []
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    sources = metadata.get("source_path") or []
    if isinstance(sources, str):
        sources = [sources]
    for source in sources:
        path = Path(source)
        if not path.exists():
            continue
        lines: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                text = raw.strip()
                if text:
                    lines.append(text)
                if len(lines) >= limit:
                    break
        return lines
    return []


REAL_LINES = _real_source_lines()


# --------------------------------------------------------------------------
# hash agreement -- the load-bearing test
# --------------------------------------------------------------------------


def test_hash64_matches_prepare_large_corpus() -> None:
    """``hash64`` must equal ``prepare_large_corpus._hash64`` byte for byte."""
    module = _load_prepare_large_corpus()
    assert module is not None, "could not import scripts/prepare_large_corpus.py"
    reference = module._hash64
    samples = [
        "",
        "[*]CC[*]",
        "[At]CCO[At]",
        "[At][C][C][O][At]",
        "c1ccccc1",
        "unicode \u00e9\u00fc\u4e2d\u6587",
        "  leading and trailing  ",
        "x" * 1000,
    ]
    for text in samples:
        assert hash64(text) == reference(text), text


def test_hash64_known_digests() -> None:
    """Pin the hash to hard-coded blake2b-64 big-endian digests.

    Guards against a future edit to *either* implementation.
    """
    expected = {
        "": 0xE4A6A057_7479B2B4,
        "[*]CC[*]": 0x313991E7_765043AE,
        "[At]CCO[At]": 0x7C422B4F_F5DC9662,
    }
    for text, digest in expected.items():
        assert hash64(text) == digest, f"{text!r} -> {hash64(text):#018x}"


def test_hash64_many_is_uint64_and_matches_scalar() -> None:
    """The vectorized hash must agree with the scalar one, as uint64."""
    codes = hash64_many(TOY_PSMILES)
    assert codes.dtype == np.uint64
    assert codes.shape == (len(TOY_PSMILES),)
    for text, code in zip(TOY_PSMILES, codes, strict=True):
        assert int(code) == hash64(text)


def test_hash64_many_empty() -> None:
    """Hashing nothing yields an empty uint64 array, not an error."""
    codes = hash64_many([])
    assert codes.dtype == np.uint64
    assert codes.size == 0


# --------------------------------------------------------------------------
# build + membership
# --------------------------------------------------------------------------


def test_from_strings_membership_exact_space() -> None:
    """Every inserted string is found; unrelated strings are not."""
    index = ScalableNoveltyIndex.from_strings(TOY_PSMILES, hash_space="psmiles")
    assert len(index) == len(TOY_PSMILES)
    for text in TOY_PSMILES:
        assert text in index
        assert not index.is_novel(text)
    for text in UNRELATED_PSMILES:
        assert text not in index
        assert index.is_novel(text)


def test_from_strings_canonical_space_collapses_writings() -> None:
    """A canonical-space index matches an equivalent writing of the polymer."""
    index = ScalableNoveltyIndex.from_strings(
        ["[*]OCC[*]"], hash_space="canonical_psmiles"
    )
    assert "[At]CCO[At]" in index
    assert "[*]CCO[*]" in index
    assert "[*]CCCCN[*]" not in index


def test_duplicate_inputs_collapse() -> None:
    """Duplicates collapse; ``len`` counts unique hashes."""
    strings = TOY_PSMILES + TOY_PSMILES + [TOY_PSMILES[0]] * 5
    index = ScalableNoveltyIndex.from_strings(strings, hash_space="psmiles")
    assert len(index) == len(set(TOY_PSMILES))
    assert index.metadata.n_entries == len(index)


def test_empty_index_is_total() -> None:
    """An empty index has length 0 and calls everything novel."""
    index = ScalableNoveltyIndex.from_strings([], hash_space="psmiles")
    assert len(index) == 0
    assert "[*]CC[*]" not in index
    assert index.is_novel("[*]CC[*]")
    mask = index.novelty_mask(TOY_PSMILES)
    assert mask.dtype == np.bool_
    assert mask.all()
    assert index.novelty_rate(TOY_PSMILES) == 1.0


def test_adversarial_input_never_raises() -> None:
    """Garbage queries answer ``novel`` rather than exploding."""
    index = ScalableNoveltyIndex.from_strings(
        TOY_PSMILES, hash_space="canonical_psmiles"
    )
    # 1000 carbons is far past anything the model can emit (generations are
    # capped at 200 tokens). Beyond ~3000 RDKit's SMILES writer recurses deep
    # enough to overflow the C stack and kill the process -- an upstream limit
    # that hits the set-based NoveltyIndex identically and is not ours to fix.
    junk = ["", "!!!not a molecule!!!", "[At][At][At", "\x00\x01", "C" * 1000]
    mask = index.novelty_mask(junk)
    assert mask.all()
    for text in junk:
        assert index.is_novel(text)


def test_from_hashes_sorts_and_dedups() -> None:
    """``from_hashes`` accepts an unsorted array with duplicates."""
    raw = np.array([9, 3, 3, 7, 1, 9], dtype=np.uint64)
    metadata = NoveltyIndexMetadata(
        n_entries=0, hash_space="psmiles", source="unit-test"
    )
    index = ScalableNoveltyIndex.from_hashes(raw, metadata=metadata)
    assert len(index) == 4
    assert index.metadata.n_entries == 4
    assert index.metadata.hash_name == HASH_NAME


# --------------------------------------------------------------------------
# vectorization
# --------------------------------------------------------------------------


def test_contains_many_agrees_with_scalar_path() -> None:
    """The vectorized path must agree element-wise with ``__contains__``."""
    index = ScalableNoveltyIndex.from_strings(
        TOY_PSMILES, hash_space="canonical_psmiles"
    )
    queries = TOY_PSMILES + UNRELATED_PSMILES + ["", "nonsense"]
    found = index.contains_many(queries)
    assert found.dtype == np.bool_
    assert found.shape == (len(queries),)
    scalar = np.array([text in index for text in queries], dtype=bool)
    assert np.array_equal(found, scalar)
    assert np.array_equal(index.novelty_mask(queries), ~scalar)
    assert index.novelty_rate(queries) == pytest.approx(float((~scalar).mean()))


def test_already_canonical_skips_canonicalization() -> None:
    """The RL fast path agrees with the slow one on genuinely canonical input."""
    from polyt5.chemistry.canonicalization import canonical_psmiles

    index = ScalableNoveltyIndex.from_strings(
        TOY_PSMILES, hash_space="canonical_psmiles"
    )
    queries = TOY_PSMILES + UNRELATED_PSMILES
    canonical = [canonical_psmiles(text) for text in queries]
    assert all(text is not None for text in canonical)
    assert np.array_equal(
        index.contains_many(canonical, already_canonical=True),
        index.contains_many(queries),
    )
    # A caller that lies gets "novel", never a wrong hit: [*] is not canonical.
    assert not index.contains_many(["[*]CC[*]"], already_canonical=True).any()


def test_contains_many_accepts_a_generator() -> None:
    """Any iterable of strings works, not just a list."""
    index = ScalableNoveltyIndex.from_strings(TOY_PSMILES, hash_space="psmiles")
    found = index.contains_many(text for text in TOY_PSMILES)
    assert found.all()


def test_contains_many_empty_query() -> None:
    """Querying nothing returns an empty bool array."""
    index = ScalableNoveltyIndex.from_strings(TOY_PSMILES, hash_space="psmiles")
    found = index.contains_many([])
    assert found.dtype == np.bool_
    assert found.size == 0
    assert index.novelty_rate([]) == 0.0


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def test_save_open_round_trip(tmp_path: Path) -> None:
    """Saving and reopening preserves membership and metadata."""
    index = ScalableNoveltyIndex.from_strings(
        TOY_PSMILES, hash_space="canonical_psmiles", source="toy"
    )
    out = tmp_path / "toy_index"
    saved = index.save(out)
    assert saved.exists()
    assert (tmp_path / "toy_index.u64").exists()
    assert (tmp_path / "toy_index.json").exists()

    reopened = ScalableNoveltyIndex.open(out)
    assert len(reopened) == len(index)
    assert reopened.metadata == index.metadata
    assert reopened.metadata.source == "toy"
    assert reopened.metadata.hash_name == HASH_NAME
    assert reopened.metadata.dtype == "uint64"
    for text in TOY_PSMILES:
        assert text in reopened
    for text in UNRELATED_PSMILES:
        assert text not in reopened


def test_open_accepts_the_u64_path(tmp_path: Path) -> None:
    """``open`` works whether given the stem or the ``.u64`` file."""
    index = ScalableNoveltyIndex.from_strings(TOY_PSMILES, hash_space="psmiles")
    index.save(tmp_path / "idx")
    reopened = ScalableNoveltyIndex.open(tmp_path / "idx.u64")
    assert len(reopened) == len(index)
    assert TOY_PSMILES[0] in reopened


def test_on_disk_array_is_sorted_and_unique(tmp_path: Path) -> None:
    """The persisted array must be strictly increasing -- binary search needs it."""
    strings = [f"[*]C{'C' * i}[*]" for i in range(200)] * 3
    index = ScalableNoveltyIndex.from_strings(strings, hash_space="psmiles")
    index.save(tmp_path / "idx")
    on_disk = np.fromfile(tmp_path / "idx.u64", dtype=np.uint64)
    assert on_disk.size == len(index)
    assert np.all(on_disk[1:] > on_disk[:-1])


def test_saved_metadata_json_is_readable(tmp_path: Path) -> None:
    """The sidecar JSON documents the index for a human and for a reload."""
    index = ScalableNoveltyIndex.from_strings(
        TOY_PSMILES, hash_space="pselfies", source="src.txt"
    )
    index.save(tmp_path / "idx")
    payload = json.loads((tmp_path / "idx.json").read_text(encoding="utf-8"))
    assert payload["hash_space"] == "pselfies"
    assert payload["hash_name"] == HASH_NAME
    assert payload["n_entries"] == len(index)
    assert payload["dtype"] == "uint64"
    assert payload["source"] == "src.txt"
    assert payload["created_utc"]


def test_save_open_empty_index(tmp_path: Path) -> None:
    """An empty index survives a round trip."""
    index = ScalableNoveltyIndex.from_strings([], hash_space="psmiles")
    index.save(tmp_path / "empty")
    reopened = ScalableNoveltyIndex.open(tmp_path / "empty")
    assert len(reopened) == 0
    assert reopened.is_novel("[*]CC[*]")


def test_open_missing_index_raises(tmp_path: Path) -> None:
    """A missing index is a caller error, surfaced clearly."""
    with pytest.raises(FileNotFoundError):
        ScalableNoveltyIndex.open(tmp_path / "nope")


# --------------------------------------------------------------------------
# hash-space guard
# --------------------------------------------------------------------------


def test_declared_hash_space_mismatch_raises() -> None:
    """Declaring a different space must raise, never answer."""
    index = ScalableNoveltyIndex.from_strings(
        TOY_PSMILES, hash_space="canonical_psmiles"
    )
    with pytest.raises(HashSpaceMismatchError) as excinfo:
        index.contains_many(TOY_PSMILES, hash_space="pselfies")
    assert "canonical_psmiles" in str(excinfo.value)
    assert "pselfies" in str(excinfo.value)
    with pytest.raises(HashSpaceMismatchError):
        index.is_novel(TOY_PSMILES[0], hash_space="pselfies")
    with pytest.raises(HashSpaceMismatchError):
        index.novelty_mask(TOY_PSMILES, hash_space="psmiles")


def test_pselfies_index_refuses_the_psmiles_surface() -> None:
    """``x in index`` carries a PSMILES contract; a PSELFIES index refuses it."""
    index = ScalableNoveltyIndex.from_strings(
        ["[At][C][C][O][At]"], hash_space="pselfies"
    )
    with pytest.raises(HashSpaceMismatchError):
        _ = "[At]CCO[At]" in index
    with pytest.raises(HashSpaceMismatchError):
        index.is_novel("[At]CCO[At]")
    # The explicit route works.
    assert index.contains_many(["[At][C][C][O][At]"], hash_space="pselfies").all()
    assert bool(index.contains_many(["[At][C][C][O][At]"])[0])


def test_unknown_hash_space_rejected() -> None:
    """Only the three declared spaces exist."""
    with pytest.raises(ValueError):
        NoveltyIndexMetadata(n_entries=0, hash_space="smiles")
    with pytest.raises(ValueError):
        ScalableNoveltyIndex.from_strings([], hash_space="not-a-space")
    assert set(HASH_SPACES) == {"pselfies", "canonical_psmiles", "psmiles"}


# --------------------------------------------------------------------------
# process safety
# --------------------------------------------------------------------------

_POOL_STATE: dict[str, ScalableNoveltyIndex] = {}


def _pool_init(index: ScalableNoveltyIndex) -> None:
    """Pool initializer: receive the pickled index and keep it per worker."""
    _POOL_STATE["index"] = index


def _pool_query(texts: list[str]) -> list[bool]:
    """Pool task: answer membership for a batch of candidates."""
    return [bool(value) for value in _POOL_STATE["index"].contains_many(texts)]


def test_index_is_safe_across_spawned_workers(tmp_path: Path) -> None:
    """A memmapped index gives identical answers from a 2-worker pool."""
    index = ScalableNoveltyIndex.from_strings(
        TOY_PSMILES, hash_space="canonical_psmiles"
    )
    index.save(tmp_path / "idx")
    opened = ScalableNoveltyIndex.open(tmp_path / "idx")

    queries = TOY_PSMILES + UNRELATED_PSMILES
    expected = [bool(value) for value in opened.contains_many(queries)]

    context = mp.get_context("spawn")
    with context.Pool(2, initializer=_pool_init, initargs=(opened,)) as pool:
        results = pool.map(_pool_query, [queries, queries, queries, queries])
    for result in results:
        assert result == expected
    # The parent's own handle still works after the children are gone.
    assert [bool(v) for v in opened.contains_many(queries)] == expected


def test_module_does_not_import_torch() -> None:
    """The reward workers are CPU-only; this module must stay torch-free."""
    code = (
        "import sys; import polyt5.chemistry.scalable_novelty as m; "
        "leaked = sorted(k for k in sys.modules if k == 'torch' or "
        "k.startswith('torch.')); "
        "assert not leaked, leaked; print(len(m.HASH_SPACES))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "3"


# --------------------------------------------------------------------------
# compatibility with the set-based index
# --------------------------------------------------------------------------


@pytest.mark.chem
@pytest.mark.skipif(not REAL_LINES, reason="dev polymer corpus not available")
def test_agrees_with_set_based_novelty_index() -> None:
    """Both indexes must return identical verdicts on real polymer strings."""
    reference = REAL_LINES[:200]
    held_out = REAL_LINES[200:300]
    assert held_out, "need held-out lines to make the test meaningful"

    legacy = NoveltyIndex(reference)
    scalable = ScalableNoveltyIndex.from_strings(
        reference, hash_space="canonical_psmiles"
    )
    assert len(scalable) == len(legacy)

    queries = reference + held_out + UNRELATED_PSMILES + ["", "junk"]
    legacy_verdicts = [legacy.is_novel(text) for text in queries]
    scalable_verdicts = [scalable.is_novel(text) for text in queries]
    assert scalable_verdicts == legacy_verdicts
    assert np.array_equal(
        scalable.novelty_mask(queries), np.array(legacy_verdicts, dtype=bool)
    )
    assert scalable.novelty_rate(queries) == pytest.approx(
        novelty_rate(queries, legacy)
    )


@pytest.mark.chem
def test_free_novelty_rate_helper_accepts_the_scalable_index() -> None:
    """``polyt5.chemistry.novelty.novelty_rate`` is duck-typed on ``is_novel``."""
    index = ScalableNoveltyIndex.from_strings(
        TOY_PSMILES, hash_space="canonical_psmiles"
    )
    queries = TOY_PSMILES + UNRELATED_PSMILES
    assert novelty_rate(queries, index) == pytest.approx(
        len(UNRELATED_PSMILES) / len(queries)
    )


@pytest.mark.chem
def test_drop_in_for_apply_filter_cascade() -> None:
    """``apply_filter_cascade`` must charge the same candidates to TSD."""
    filters = pytest.importorskip("polyt5.evaluation.filters")
    from polyt5.chemistry.conversion import psmiles_to_pselfies

    pselfies = [psmiles_to_pselfies(text) for text in TOY_PSMILES + UNRELATED_PSMILES]
    assert all(value is not None for value in pselfies)

    known = TOY_PSMILES
    legacy = NoveltyIndex(known)
    scalable = ScalableNoveltyIndex.from_strings(known, hash_space="canonical_psmiles")

    legacy_records, legacy_counts = filters.apply_filter_cascade(
        pselfies, training_index=legacy
    )
    scalable_records, scalable_counts = filters.apply_filter_cascade(
        pselfies, training_index=scalable
    )
    assert scalable_counts == legacy_counts
    assert [r.failure_stage for r in scalable_records] == [
        r.failure_stage for r in legacy_records
    ]
    assert legacy_counts.n_tsd == len(UNRELATED_PSMILES)


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------


def _dev_sidecar() -> Path:
    """Return the path of the dev corpus dedup sidecar."""
    return POLYONE_DEV_PREFIX.with_name(POLYONE_DEV_PREFIX.name + ".dedup.u64")


@pytest.mark.slow
@pytest.mark.skipif(not _dev_sidecar().exists(), reason="dev dedup sidecar not built")
def test_from_dedup_sidecar_real_corpus() -> None:
    """Build from the real ``.dedup.u64`` and answer real queries correctly."""
    index = ScalableNoveltyIndex.from_dedup_sidecar(POLYONE_DEV_PREFIX)
    assert len(index) > 1_000_000
    assert index.metadata.hash_space in HASH_SPACES
    assert index.metadata.hash_name == HASH_NAME
    assert str(POLYONE_DEV_PREFIX.name) in index.metadata.source

    lines = _real_source_lines(limit=64)
    assert lines, "the corpus source file must be present for this assertion"
    found = index.contains_many(lines, hash_space=index.metadata.hash_space)
    # A few source rows are dropped by the length filter and are legitimately
    # absent; the overwhelming majority must be present.
    assert found.mean() > 0.9, f"only {found.mean():.3f} of source lines found"

    randoms = [f"[*]C{'C' * i}N(CC)c1ccccc1Br{i}[*]" for i in range(64)]
    assert not index.contains_many(
        randoms, hash_space=index.metadata.hash_space
    ).any()


@pytest.mark.slow
@pytest.mark.skipif(not _dev_sidecar().exists(), reason="dev dedup sidecar not built")
def test_dedup_sidecar_hash_space_conflict_raises() -> None:
    """Asking for a space the corpus was not deduplicated in must raise."""
    metadata = json.loads(
        POLYONE_DEV_PREFIX.with_name(POLYONE_DEV_PREFIX.name + ".json").read_text(
            encoding="utf-8"
        )
    )
    corpus_space = {"exact": "psmiles", "canonical": "canonical_psmiles"}[
        metadata["dedup"]
    ]
    wrong = "pselfies" if corpus_space != "pselfies" else "psmiles"
    with pytest.raises(HashSpaceMismatchError):
        ScalableNoveltyIndex.from_dedup_sidecar(POLYONE_DEV_PREFIX, hash_space=wrong)


# --------------------------------------------------------------------------
# builder script
# --------------------------------------------------------------------------


def test_build_script_from_text(tmp_path: Path) -> None:
    """The CLI builds a queryable index from a text file."""
    source = tmp_path / "polymers.txt"
    source.write_text("\n".join(TOY_PSMILES) + "\n", encoding="utf-8")
    out = tmp_path / "cli_index"
    result = subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--from-text",
            str(source),
            "--hash-space",
            "psmiles",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    index = ScalableNoveltyIndex.open(out)
    assert len(index) == len(TOY_PSMILES)
    for text in TOY_PSMILES:
        assert text in index


def test_build_script_parallel_and_limited(tmp_path: Path) -> None:
    """``--workers 2 --limit N`` hashes in a pool and truncates the stream."""
    strings = [f"[*]C{'C' * i}[*]" for i in range(64)]
    source = tmp_path / "many.txt"
    source.write_text("\n".join(strings) + "\n", encoding="utf-8")
    out = tmp_path / "parallel_index"
    result = subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--from-text",
            str(source),
            "--hash-space",
            "psmiles",
            "--workers",
            "2",
            "--limit",
            "10",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    index = ScalableNoveltyIndex.open(out)
    assert len(index) == 10
    assert strings[0] in index
    assert strings[10] not in index


def test_build_script_rejects_a_lying_hash_space(tmp_path: Path) -> None:
    """``--canonicalize`` without the canonical space is refused, not honoured."""
    source = tmp_path / "polymers.txt"
    source.write_text("\n".join(TOY_PSMILES) + "\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--from-text",
            str(source),
            "--hash-space",
            "psmiles",
            "--canonicalize",
            "--out",
            str(tmp_path / "bad"),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 2
    assert "canonical_psmiles" in result.stderr
    assert not (tmp_path / "bad.u64").exists()


def test_build_script_from_jsonl_with_canonicalization(tmp_path: Path) -> None:
    """``--from-jsonl --canonicalize`` builds a canonical-space index."""
    source = tmp_path / "rows.jsonl"
    source.write_text(
        "\n".join(json.dumps({"target": text}) for text in TOY_PSMILES) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "jsonl_index"
    result = subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--from-jsonl",
            str(source),
            "--field",
            "target",
            "--hash-space",
            "canonical_psmiles",
            "--canonicalize",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    index = ScalableNoveltyIndex.open(out)
    assert index.metadata.hash_space == "canonical_psmiles"
    assert "[At]CCO[At]" in index


# ---------------------------------------------------------------------------
# self_check: the empirical guard against an undeclared hash-space mismatch
# ---------------------------------------------------------------------------


def test_self_check_passes_for_members_in_the_right_space():
    members = ["[At]CC[At]", "[At]CCO[At]", "[At]CC(C)[At]"]
    index = ScalableNoveltyIndex.from_strings(members, hash_space="psmiles")
    assert index.self_check(members) == 1.0


def test_self_check_catches_a_wrong_space_query():
    """The failure this method exists to catch.

    Nothing in Python distinguishes a PSELFIES string from a PSMILES string by
    type, so an undeclared query in the wrong space is silently answered
    "everything is novel" — which looks like an excellent novelty result rather
    than a bug. This is exactly how a real index built from a corpus dedup
    sidecar reported 10,000 of 10,000 known polymers as novel.
    """
    psmiles = ["[At]CC[At]", "[At]CCO[At]", "[At]CC(C)[At]"]
    pselfies = ["[At][C][C][At]", "[At][C][C][O][At]", "[At][C][C][Branch1][C][C][At]"]
    index = ScalableNoveltyIndex.from_strings(psmiles, hash_space="psmiles")

    with pytest.raises(HashSpaceMismatchError, match="self-check failed"):
        index.self_check(pselfies)


def test_self_check_honours_min_hit_rate():
    """A sample containing known non-members is tolerated when asked for."""
    members = ["[At]CC[At]", "[At]CCO[At]", "[At]CC(C)[At]", "[At]CCCC[At]"]
    index = ScalableNoveltyIndex.from_strings(members, hash_space="psmiles")
    sample = [*members, "[At]C(F)(F)C(F)(F)[At]"]  # 4 of 5 present

    assert index.self_check(sample, min_hit_rate=0.75) == pytest.approx(0.8)
    with pytest.raises(HashSpaceMismatchError):
        index.self_check(sample, min_hit_rate=1.0)


def test_self_check_on_empty_sample_is_vacuously_true():
    index = ScalableNoveltyIndex.from_strings(["[At]CC[At]"], hash_space="psmiles")
    assert index.self_check([]) == 1.0


# ---------------------------------------------------------------------------
# build script: the drop-rate guard and PSELFIES input
# ---------------------------------------------------------------------------


def _write_pselfies_jsonl(path: Path, pselfies: list[str]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for text in pselfies:
            fh.write(json.dumps({"source": "400.0", "target": text}) + "\n")


PSELFIES_ROWS = [
    "[At][C][C][At]",
    "[At][C][C][O][At]",
    "[At][C][C][Branch1][C][C][At]",
    "[At][C][C][Branch1][C][Cl][At]",
    "[At][C][C][C][C][At]",
]


def test_build_script_rejects_a_space_mismatch_instead_of_writing_an_empty_index(tmp_path):
    """Feeding PSELFIES to a canonical_psmiles build must fail loudly.

    Canonicalizing a PSELFIES string returns None, so every row is discarded and
    the script would otherwise write a near-empty index that reports every future
    candidate as novel — a broken build that looks like a great result.
    """
    source = tmp_path / "rows.jsonl"
    _write_pselfies_jsonl(source, PSELFIES_ROWS)

    result = subprocess.run(
        [
            sys.executable, str(BUILD_SCRIPT),
            "--from-jsonl", str(source), "--field", "target",
            "--hash-space", "canonical_psmiles", "--canonicalize",
            "--out", str(tmp_path / "bad"),
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 1, result.stdout
    assert "could not be hashed" in result.stdout
    assert "pselfies_to_psmiles" in result.stdout


def test_build_script_converts_pselfies_when_asked(tmp_path):
    source = tmp_path / "rows.jsonl"
    _write_pselfies_jsonl(source, PSELFIES_ROWS)
    out = tmp_path / "good"

    result = subprocess.run(
        [
            sys.executable, str(BUILD_SCRIPT),
            "--from-jsonl", str(source), "--field", "target",
            "--hash-space", "canonical_psmiles", "--canonicalize", "--from-pselfies",
            "--out", str(out),
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr + result.stdout

    index = ScalableNoveltyIndex.open(out)
    assert len(index) == len(PSELFIES_ROWS)
    # every source row must be findable through its PSMILES form
    known = [pselfies_to_psmiles(t) for t in PSELFIES_ROWS]
    assert index.self_check([k for k in known if k]) == 1.0


def test_build_script_allows_expected_attrition_when_told(tmp_path):
    """--max-drop-rate 1.0 is the documented escape hatch."""
    source = tmp_path / "rows.jsonl"
    _write_pselfies_jsonl(source, PSELFIES_ROWS)

    result = subprocess.run(
        [
            sys.executable, str(BUILD_SCRIPT),
            "--from-jsonl", str(source), "--field", "target",
            "--hash-space", "canonical_psmiles", "--canonicalize",
            "--max-drop-rate", "1.0",
            "--out", str(tmp_path / "permitted"),
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout
