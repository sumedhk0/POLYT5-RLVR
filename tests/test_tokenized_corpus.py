"""Tests for the memory-mapped, pre-tokenized corpus and its parallel builder.

Why this layer exists (measured, not assumed):

* ``scripts/prepare_pretraining_data.py`` converts PSMILES -> PSELFIES on one
  core at ~6.3k rows/s; 100M rows is 4.4 hours of wall clock.
* ``PSelfiesCorpus`` tokenizes inside ``__getitem__`` from a Python list held
  in RAM. In the real polyT5-small run epoch 0 took 5,215 s and epoch 1 took
  270 s -- a 19x gap that is pure per-item Python overhead -- and 100M PSELFIES
  strings do not fit in RAM at all.
* Tokenization itself is NOT the bottleneck: 77.7k seq/s on one core.

So the corpus is tokenized once, in parallel, into a flat uint16 ``.bin`` plus a
uint64 ``.idx``, and every DataLoader worker mmaps it read-only.

The load-bearing test in this file is
:func:`test_dataloader_workers_match_single_process`: an ``np.memmap`` held
across a fork/spawn is the classic way this design breaks, and the failure is
silent (wrong or zero-filled data), not loud.

Nothing here touches the network or the 8 GB polyOne download.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from polyt5.data.collate import LABEL_IGNORE_ID, SpanCorruptionCollator  # noqa: E402
from polyt5.data.span_corruption import SpanCorruptionConfig  # noqa: E402
from polyt5.data.tokenized_corpus import (  # noqa: E402
    CORPUS_FORMAT_VERSION,
    MemmapPSelfiesDataset,
    TokenizedCorpus,
    TokenizedCorpusWriter,
    TokenizerMismatchError,
    load_split_indices,
    verify_corpus,
)
from polyt5.tokenization import PolyT5Tokenizer  # noqa: E402

TOKENIZER_PATH = REPO_ROOT / "artifacts" / "tokenizer" / "polyt5_vocab.json"
PREPARE_SCRIPT = REPO_ROOT / "scripts" / "prepare_large_corpus.py"

# A sha256 that is a well-formed hex digest but belongs to no real vocabulary.
FOREIGN_SHA256 = "0" * 64


@pytest.fixture(scope="module")
def tokenizer() -> PolyT5Tokenizer:
    """The real 458-token vocabulary artifact."""
    if not TOKENIZER_PATH.exists():
        pytest.skip(f"tokenizer artifact missing: {TOKENIZER_PATH}")
    return PolyT5Tokenizer.from_file(TOKENIZER_PATH)


def _random_sequences(n: int, *, seed: int = 0, vocab: int = 458) -> list[list[int]]:
    """Build ``n`` variable-length id sequences with a deterministic RNG."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 60, size=n)
    return [rng.integers(0, vocab, size=int(k)).tolist() for k in lengths]


def _identity_collate(batch: list[list[int]]) -> list[list[int]]:
    """Pass rows through untouched (must be module level: Windows uses spawn)."""
    return batch


def _write(prefix: Path, sequences, **kwargs) -> Path:
    """Write ``sequences`` to a corpus at ``prefix`` and return the prefix."""
    kwargs.setdefault("tokenizer_sha256", "a" * 64)
    kwargs.setdefault("vocab_size", 458)
    with TokenizedCorpusWriter(prefix, **kwargs) as writer:
        writer.add_many(sequences)
    return prefix


# --------------------------------------------------------------------------
# writer / reader round trip
# --------------------------------------------------------------------------


def test_round_trip_preserves_every_sequence(tmp_path: Path) -> None:
    sequences = _random_sequences(500, seed=1)
    prefix = _write(tmp_path / "corpus", sequences)

    corpus = TokenizedCorpus.from_prefix(prefix)
    assert len(corpus) == 500
    for i, expected in enumerate(sequences):
        assert corpus[i] == expected, f"sequence {i} did not round trip"


def test_offsets_handle_extreme_lengths(tmp_path: Path) -> None:
    """A length-1 and a length-200 sequence must survive next to normal rows."""
    sequences = [
        [7],
        list(range(200)),
        [1, 2, 3],
        [457],
        list(range(100, 300)),  # another 200-long row
    ]
    prefix = _write(tmp_path / "corpus", sequences, max_length=200)

    corpus = TokenizedCorpus.from_prefix(prefix)
    assert [len(corpus[i]) for i in range(len(corpus))] == [1, 200, 3, 1, 200]
    for i, expected in enumerate(sequences):
        assert corpus[i] == expected

    offsets = np.fromfile(Path(str(prefix) + ".idx"), dtype=np.uint64)
    assert offsets.shape == (len(sequences) + 1,)
    assert offsets[0] == 0
    assert np.all(np.diff(offsets.astype(np.int64)) == np.array([1, 200, 3, 1, 200]))
    assert int(offsets[-1]) == corpus.n_tokens == sum(len(s) for s in sequences)


def test_negative_indexing_and_bounds(tmp_path: Path) -> None:
    sequences = [[1, 2], [3], [4, 5, 6]]
    prefix = _write(tmp_path / "corpus", sequences)
    corpus = TokenizedCorpus.from_prefix(prefix)

    assert corpus[-1] == [4, 5, 6]
    assert corpus[-3] == [1, 2]
    with pytest.raises(IndexError):
        corpus[3]
    with pytest.raises(IndexError):
        corpus[-4]


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------


def test_metadata_round_trip(tmp_path: Path) -> None:
    stats = {"n_input": 10, "n_kept": 7, "n_duplicate": 3}
    prefix = tmp_path / "corpus"
    with TokenizedCorpusWriter(
        prefix,
        tokenizer_sha256="b" * 64,
        tokenizer_path="artifacts/tokenizer/polyt5_vocab.json",
        max_length=200,
        vocab_size=458,
        source_path=["raw_a.txt", "raw_b.txt"],
    ) as writer:
        writer.set_preparation_stats(stats)
        writer.add_many([[1, 2, 3], [4]])

    meta = json.loads(Path(str(prefix) + ".json").read_text(encoding="utf-8"))
    assert meta["format_version"] == CORPUS_FORMAT_VERSION
    assert meta["n_sequences"] == 2
    assert meta["n_tokens"] == 4
    assert meta["dtype"] == "uint16"
    assert meta["max_length"] == 200
    assert meta["tokenizer_sha256"] == "b" * 64
    assert meta["tokenizer_path"] == "artifacts/tokenizer/polyt5_vocab.json"
    assert meta["source_path"] == ["raw_a.txt", "raw_b.txt"]
    assert meta["preparation_stats"] == stats
    assert meta["created_utc"].endswith("+00:00")
    assert meta["polyt5_version"]

    corpus = TokenizedCorpus.from_prefix(prefix)
    assert corpus.metadata == meta
    assert corpus.tokenizer_sha256 == "b" * 64
    assert corpus.n_tokens == 4


# --------------------------------------------------------------------------
# degenerate corpora
# --------------------------------------------------------------------------


def test_empty_corpus(tmp_path: Path) -> None:
    prefix = _write(tmp_path / "corpus", [])
    corpus = TokenizedCorpus.from_prefix(prefix)

    assert len(corpus) == 0
    assert corpus.n_tokens == 0
    assert list(corpus) == []
    with pytest.raises(IndexError):
        corpus[0]

    dataset = MemmapPSelfiesDataset(corpus)
    assert len(dataset) == 0


def test_single_sequence_corpus(tmp_path: Path) -> None:
    prefix = _write(tmp_path / "corpus", [[5, 6, 7]])
    corpus = TokenizedCorpus.from_prefix(prefix)

    assert len(corpus) == 1
    assert corpus[0] == [5, 6, 7]
    assert corpus.n_tokens == 3


def test_empty_sequence_is_rejected(tmp_path: Path) -> None:
    """A zero-length row would produce a loss-free batch row; refuse it early."""
    with pytest.raises(ValueError, match="empty"):
        with TokenizedCorpusWriter(
            tmp_path / "corpus", tokenizer_sha256="c" * 64, vocab_size=458
        ) as writer:
            writer.add([])


# --------------------------------------------------------------------------
# dtype guard
# --------------------------------------------------------------------------


def test_uint16_rejects_oversized_vocabulary(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        TokenizedCorpusWriter(
            tmp_path / "corpus",
            tokenizer_sha256="d" * 64,
            vocab_size=70_000,
            dtype="uint16",
        )


def test_uint32_accepts_oversized_vocabulary(tmp_path: Path) -> None:
    prefix = tmp_path / "corpus"
    with TokenizedCorpusWriter(
        prefix, tokenizer_sha256="d" * 64, vocab_size=70_000, dtype="uint32"
    ) as writer:
        writer.add([69_999, 0, 12])

    corpus = TokenizedCorpus.from_prefix(prefix)
    assert corpus[0] == [69_999, 0, 12]
    assert corpus.metadata["dtype"] == "uint32"


def test_out_of_range_token_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="out of range"):
        with TokenizedCorpusWriter(
            tmp_path / "corpus", tokenizer_sha256="d" * 64, vocab_size=458
        ) as writer:
            writer.add([1, 2, 70_000])


# --------------------------------------------------------------------------
# verify_corpus
# --------------------------------------------------------------------------


def test_verify_corpus_reports_statistics(tmp_path: Path, tokenizer: PolyT5Tokenizer) -> None:
    # Ids 10+ are ordinary vocabulary entries, never <unk>.
    sequences = [[10], list(range(10, 210)), [11, 12, 13, 14]]
    prefix = _write(tmp_path / "corpus", sequences, tokenizer_sha256=tokenizer.sha256)

    report = verify_corpus(prefix, tokenizer)
    assert report["n_sequences"] == 3
    assert report["n_tokens"] == 205
    assert report["min_length"] == 1
    assert report["max_length"] == 200
    assert report["tokenizer_sha256_matches"] is True
    assert report["n_unknown_tokens"] == 0
    assert report["ok"] is True


def test_verify_corpus_raises_on_tokenizer_mismatch(
    tmp_path: Path, tokenizer: PolyT5Tokenizer
) -> None:
    prefix = _write(tmp_path / "corpus", [[1, 2, 3]], tokenizer_sha256=FOREIGN_SHA256)

    with pytest.raises(TokenizerMismatchError) as excinfo:
        verify_corpus(prefix, tokenizer)
    message = str(excinfo.value)
    assert FOREIGN_SHA256[:12] in message
    assert tokenizer.sha256[:12] in message


def test_verify_corpus_counts_unknown_tokens(tmp_path: Path, tokenizer: PolyT5Tokenizer) -> None:
    prefix = _write(
        tmp_path / "corpus",
        [[tokenizer.unk_id, 5, tokenizer.unk_id], [6, 7]],
        tokenizer_sha256=tokenizer.sha256,
    )
    report = verify_corpus(prefix, tokenizer)
    assert report["n_unknown_tokens"] == 2


def test_verify_corpus_detects_truncated_bin(tmp_path: Path, tokenizer: PolyT5Tokenizer) -> None:
    prefix = _write(tmp_path / "corpus", _random_sequences(20), tokenizer_sha256=tokenizer.sha256)
    bin_path = Path(str(prefix) + ".bin")
    with bin_path.open("r+b") as fh:
        fh.truncate(bin_path.stat().st_size - 8)

    with pytest.raises(ValueError, match="truncated|size"):
        verify_corpus(prefix, tokenizer)


# --------------------------------------------------------------------------
# views / dataset / collator
# --------------------------------------------------------------------------


def test_slice_view_exposes_a_split_without_copying(tmp_path: Path) -> None:
    sequences = _random_sequences(50, seed=2)
    prefix = _write(tmp_path / "corpus", sequences)
    corpus = TokenizedCorpus.from_prefix(prefix)

    indices = [7, 0, 49, 23]
    view = corpus.slice_view(indices)
    assert len(view) == 4
    assert view[0] == sequences[7]
    assert view[2] == sequences[49]
    assert list(view) == [sequences[i] for i in indices]


def test_dataset_and_collator_produce_a_valid_batch(
    tmp_path: Path, tokenizer: PolyT5Tokenizer
) -> None:
    # Long enough that span corruption always finds room -- a 4-token row
    # legitimately yields all-(-100) labels and would not test anything.
    pselfies = ["[At]" + "[C]" * k + "[At]" for k in range(20, 52)]
    encoded = [tokenizer.encode(p, add_eos=True, max_length=200, truncation=True) for p in pselfies]
    prefix = _write(tmp_path / "corpus", encoded, tokenizer_sha256=tokenizer.sha256)

    corpus = TokenizedCorpus.from_prefix(prefix)
    dataset = MemmapPSelfiesDataset(corpus, list(range(len(corpus))), max_length=200)
    collator = SpanCorruptionCollator(
        sentinel_ids=tokenizer.sentinel_ids,
        pad_id=tokenizer.pad_id,
        eos_id=tokenizer.eos_id,
        config=SpanCorruptionConfig(),
        max_length=200,
        seed=0,
    )

    item = dataset[0]
    assert isinstance(item, list)
    assert all(isinstance(x, int) for x in item)

    batch = collator([dataset[i] for i in range(8)])
    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].dtype == torch.long
    assert batch["input_ids"].shape[0] == 8
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["labels"].shape[0] == 8
    # Padded label positions are -100 so the loss ignores them (paper's rule).
    row_lengths = (batch["labels"] != LABEL_IGNORE_ID).sum(dim=1)
    assert int(row_lengths.min()) > 0
    padded = batch["labels"] == LABEL_IGNORE_ID
    assert bool(padded.any()), "variable-length labels should have produced padding"


def test_dataloader_workers_match_single_process(tmp_path: Path) -> None:
    """The fork/spawn-safety test: a memmap must not be shared across processes.

    A worker that inherits an already-open ``np.memmap`` reads garbage (or, on
    Windows spawn, fails to pickle). This test is the one that catches it.
    """
    sequences = _random_sequences(64, seed=3)
    prefix = _write(tmp_path / "corpus", sequences)
    corpus = TokenizedCorpus.from_prefix(prefix)
    dataset = MemmapPSelfiesDataset(corpus)

    # Touch the memmap in the parent BEFORE spawning workers -- that is exactly
    # the state that breaks a naive implementation.
    assert dataset[0] == sequences[0]

    single = [
        row
        for batch in DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=_identity_collate)
        for row in batch
    ]
    parallel = [
        row
        for batch in DataLoader(
            dataset, batch_size=8, shuffle=False, collate_fn=_identity_collate, num_workers=2
        )
        for row in batch
    ]

    assert single == sequences
    assert parallel == single


def test_corpus_pickles_without_open_memmaps(tmp_path: Path) -> None:
    import pickle

    sequences = _random_sequences(10, seed=4)
    prefix = _write(tmp_path / "corpus", sequences)
    corpus = TokenizedCorpus.from_prefix(prefix)
    assert corpus[0] == sequences[0]  # force the memmap open

    state = corpus.__getstate__()
    assert state.get("_tokens") is None
    assert state.get("_offsets") is None

    revived = pickle.loads(pickle.dumps(corpus))
    assert revived[5] == sequences[5]


# --------------------------------------------------------------------------
# scripts/prepare_large_corpus.py
# --------------------------------------------------------------------------


def _synthetic_psmiles_lines() -> list[str]:
    """Build ~200 real, mostly-distinct PSMILES lines with known attrition.

    Returns:
        Raw star-notation lines; the composition is asserted by the callers.
    """
    lines: list[str] = []
    lines += [f"[*]{'C' * k}[*]" for k in range(1, 121)]  # 120 valid, distinct
    lines += [f"[*]O{'C' * k}[*]" for k in range(1, 61)]  # 60 valid, distinct
    duplicates = [f"[*]{'C' * k}[*]" for k in range(1, 11)]  # 10 exact duplicates
    lines += duplicates
    lines += ["[*]" + "C" * 250 + "[*]", "[*]" + "C" * 300 + "[*]"]  # 2 over 200 tokens
    lines += ["not_a_smiles!!", "&&&", "C(((C"]  # 3 unparseable
    lines += ["[*]CC", "[*]CCC"]  # 2 with one terminus
    return lines


N_VALID_UNIQUE = 180
N_DUPLICATE = 10
N_TOO_LONG = 2
N_PARSE_FAILED = 3
N_WRONG_TERMINI = 2
N_LINES = N_VALID_UNIQUE + N_DUPLICATE + N_TOO_LONG + N_PARSE_FAILED + N_WRONG_TERMINI


def _run_prepare(*args: str) -> subprocess.CompletedProcess:
    """Run ``scripts/prepare_large_corpus.py`` as a subprocess (spawn-safe)."""
    return subprocess.run(
        [sys.executable, str(PREPARE_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )


@pytest.fixture
def raw_input(tmp_path: Path) -> Path:
    path = tmp_path / "raw.txt"
    path.write_text("\n".join(_synthetic_psmiles_lines()) + "\n", encoding="utf-8")
    return path


@pytest.mark.slow
def test_prepare_large_corpus_end_to_end(
    tmp_path: Path, raw_input: Path, tokenizer: PolyT5Tokenizer
) -> None:
    prefix = tmp_path / "out" / "corpus"
    result = _run_prepare(
        "--input", str(raw_input),
        "--output-prefix", str(prefix),
        "--tokenizer", str(TOKENIZER_PATH),
        "--workers", "2",
        "--chunk-size", "50",
        "--max-length", "200",
        "--dedup", "exact",
        "--seed", "0",
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    corpus = TokenizedCorpus.from_prefix(prefix)
    assert len(corpus) == N_VALID_UNIQUE

    stats = corpus.metadata["preparation_stats"]
    assert stats["n_input"] == N_LINES
    assert stats["n_kept"] == N_VALID_UNIQUE
    assert stats["n_duplicate"] == N_DUPLICATE
    assert stats["n_too_long"] == N_TOO_LONG
    assert stats["n_parse_failed"] == N_PARSE_FAILED
    assert stats["n_wrong_termini"] == N_WRONG_TERMINI
    # The buckets must partition the input exactly.
    assert stats["n_input"] == sum(
        stats[k] for k in stats if k != "n_input"
    )

    # stats.json sidecar mirrors the metadata.
    sidecar = json.loads((prefix.parent / "stats.json").read_text(encoding="utf-8"))
    assert sidecar["attrition"] == stats
    assert sidecar["tokenizer_sha256"] == tokenizer.sha256

    # Splits partition the corpus exactly.
    splits_path = prefix.parent / "splits.json"
    payload = json.loads(splits_path.read_text(encoding="utf-8"))
    assert payload["n"] == N_VALID_UNIQUE
    parts = {name: load_split_indices(splits_path, name) for name in ("train", "val", "test")}
    combined = np.concatenate(list(parts.values()))
    assert combined.size == N_VALID_UNIQUE
    assert np.array_equal(np.unique(combined), np.arange(N_VALID_UNIQUE))
    assert len(parts["train"]) == round(0.9 * N_VALID_UNIQUE)

    # The produced corpus verifies against the real tokenizer.
    report = verify_corpus(prefix, tokenizer)
    assert report["ok"] is True
    assert report["n_unknown_tokens"] == 0
    assert report["tokenizer_sha256_matches"] is True

    # Every stored sequence decodes to a PSELFIES string that re-encodes to itself.
    for i in (0, len(corpus) // 2, len(corpus) - 1):
        text = tokenizer.decode(corpus[i])
        assert text.startswith("[At]") and text.endswith("[At]")
        assert tokenizer.encode(text, add_eos=True) == corpus[i]


@pytest.mark.slow
def test_prepare_large_corpus_resume_does_not_duplicate(
    tmp_path: Path, raw_input: Path
) -> None:
    common = [
        "--input", str(raw_input),
        "--tokenizer", str(TOKENIZER_PATH),
        "--workers", "2",
        "--chunk-size", "50",
        "--dedup", "exact",
        "--seed", "0",
    ]

    full_prefix = tmp_path / "full" / "corpus"
    assert _run_prepare(*common, "--output-prefix", str(full_prefix)).returncode == 0
    full = TokenizedCorpus.from_prefix(full_prefix)
    expected = [full[i] for i in range(len(full))]

    # Interrupt-simulate: process only the first 100 lines...
    partial_prefix = tmp_path / "partial" / "corpus"
    first = _run_prepare(*common, "--output-prefix", str(partial_prefix), "--limit", "100")
    assert first.returncode == 0
    partial = TokenizedCorpus.from_prefix(partial_prefix)
    n_partial = len(partial)
    assert 0 < n_partial < len(full)

    # ...then resume and finish the file.
    second = _run_prepare(*common, "--output-prefix", str(partial_prefix), "--resume")
    assert second.returncode == 0, f"stdout:\n{second.stdout}\nstderr:\n{second.stderr}"

    resumed = TokenizedCorpus.from_prefix(partial_prefix)
    got = [resumed[i] for i in range(len(resumed))]
    assert len(got) == len(expected), "resume produced a different number of sequences"
    assert got == expected, "resume duplicated or dropped sequences"
    assert resumed.metadata["preparation_stats"] == full.metadata["preparation_stats"]

    # Re-resuming a finished corpus is a no-op, not a doubling.
    third = _run_prepare(*common, "--output-prefix", str(partial_prefix), "--resume")
    assert third.returncode == 0
    again = TokenizedCorpus.from_prefix(partial_prefix)
    assert len(again) == len(expected)


@pytest.mark.slow
def test_prepare_large_corpus_dedup_none_keeps_duplicates(
    tmp_path: Path, raw_input: Path
) -> None:
    prefix = tmp_path / "nodedup" / "corpus"
    result = _run_prepare(
        "--input", str(raw_input),
        "--output-prefix", str(prefix),
        "--tokenizer", str(TOKENIZER_PATH),
        "--workers", "2",
        "--chunk-size", "64",
        "--dedup", "none",
    )
    assert result.returncode == 0, result.stderr
    corpus = TokenizedCorpus.from_prefix(prefix)
    assert len(corpus) == N_VALID_UNIQUE + N_DUPLICATE
    assert corpus.metadata["preparation_stats"]["n_duplicate"] == 0
