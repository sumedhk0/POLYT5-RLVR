"""Tests for the data acquisition and preparation pipeline.

Ground truth (Sahu et al., npj Artificial Intelligence 2026):

* PSMILES uses two ``[*]`` chain-end markers; ``selfies`` cannot encode ``*``,
  so ``[*]`` is replaced by ``[At]`` and SELFIES-encoded -> PSELFIES.
* Max sequence length: 200 tokens.
* Task I/O (verbatim from the SI): property prediction input is the bare
  PSELFIES, output the numeric string ``"236.0"`` (one decimal place, no task
  prefix); Tg-conditioned generation inverts that mapping.
* Splits: pretraining 90/10; property prediction 80/20 over five random
  splits; generation 90/10.

No test in this file touches the network, and no test reads more than the
first few hundred rows of the real CSVs.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from polyt5.data.prepare import (  # noqa: E402
    PreparationStats,
    build_generation_examples,
    build_prediction_examples,
    format_property_value,
    parse_property_value,
    prepare_labeled_corpus,
    prepare_pselfies_corpus,
    read_lamalab_tg,
    read_pi1m,
)
from polyt5.data.sources import (  # noqa: E402
    SOURCES,
    ConfirmationRequiredError,
    DataSource,
    DownloadRecord,
    download,
)
from polyt5.data.splits import (  # noqa: E402
    load_splits,
    make_kfold_random_splits,
    make_pretraining_splits,
    random_split,
    save_splits,
)

PI1M_PATH = REPO_ROOT / "data" / "external" / "PI1M_v2.csv"
LAMALAB_PATH = REPO_ROOT / "data" / "external" / "LAMALAB_CURATED_Tg.csv"


# ---------------------------------------------------------------------------
# Fake tokenizer (duck-typed stand-in for polyt5.tokenization.PolyT5Tokenizer,
# which is being written concurrently and must not be imported here).
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """One token id per character; picklable (module-level class, plain state)."""

    pad_id = 0
    eos_id = 1
    vocab_size = 402

    def encode(self, text, add_eos=True, max_length=None, truncation=True):
        ids = [(ord(c) % 400) + 2 for c in text]
        if add_eos:
            ids.append(self.eos_id)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return ids


# ---------------------------------------------------------------------------
# prepare_pselfies_corpus
# ---------------------------------------------------------------------------


class TestPreparePselfiesCorpus:
    def test_hand_built_rows_exercise_every_counter(self):
        rows = [
            "[*]CCO[*]",  # valid -> kept
            "*CCO",  # 1 terminus -> rejected
            "*CC(*)O*",  # 3 termini -> rejected
            "not_a_smiles",  # RDKit garbage -> rejected
            "*" + "C" * 250 + "*",  # 252 SELFIES tokens -> too long
            "*OCC*",  # same polymer as row 0 -> duplicate
        ]
        kept, stats = prepare_pselfies_corpus(rows)

        assert isinstance(stats, PreparationStats)
        assert stats.n_input == 6
        assert stats.n_parse_failed == 1
        assert stats.n_wrong_termini == 2
        assert stats.n_selfies_failed == 0
        assert stats.n_too_long == 1
        assert stats.n_duplicate == 1
        assert stats.n_kept == 1
        assert kept == ["[At][C][C][O][At]"]

    def test_stats_partition_the_input(self):
        rows = ["[*]CCO[*]", "*CCO", "junk", "*OCC*", "*C(=O)O*"]
        kept, stats = prepare_pselfies_corpus(rows)
        dropped = (
            stats.n_parse_failed
            + stats.n_wrong_termini
            + stats.n_selfies_failed
            + stats.n_too_long
            + stats.n_duplicate
        )
        assert stats.n_input == dropped + stats.n_kept
        assert len(kept) == stats.n_kept

    def test_deduplicate_false_keeps_both_notations(self):
        kept, stats = prepare_pselfies_corpus(["[*]CCO[*]", "*OCC*"], deduplicate=False)
        assert stats.n_duplicate == 0
        assert stats.n_kept == 2
        assert kept == ["[At][C][C][O][At]", "[At][O][C][C][At]"]

    def test_injected_tokenizer_controls_length_filter(self):
        # FakeTokenizer counts characters (+1 EOS). "[At][C][C][O][At]" is 17
        # chars -> 18 ids, so max_tokens=10 must drop it even though it is only
        # 5 SELFIES tokens.
        kept, stats = prepare_pselfies_corpus(
            ["[*]CCO[*]"], max_tokens=10, tokenizer=FakeTokenizer()
        )
        assert kept == []
        assert stats.n_too_long == 1
        kept, stats = prepare_pselfies_corpus(
            ["[*]CCO[*]"], max_tokens=200, tokenizer=FakeTokenizer()
        )
        assert stats.n_kept == 1

    def test_to_dict_round_trips_every_field(self):
        _, stats = prepare_pselfies_corpus(["[*]CCO[*]"])
        d = stats.to_dict()
        assert d == {
            "n_input": 1,
            "n_parse_failed": 0,
            "n_wrong_termini": 0,
            "n_selfies_failed": 0,
            "n_too_long": 0,
            "n_duplicate": 0,
            "n_kept": 1,
        }

    def test_labeled_corpus_keeps_pairing_and_stats(self):
        pairs = [("[*]CCO[*]", 236.04), ("*CCO", 100.0), ("*OCC*", 300.0)]
        kept, stats = prepare_labeled_corpus(pairs)
        assert kept == [("[At][C][C][O][At]", 236.04)]
        assert stats.n_input == 3
        assert stats.n_wrong_termini == 1
        assert stats.n_duplicate == 1
        assert stats.n_kept == 1


# ---------------------------------------------------------------------------
# Real-data smoke tests (first 200 rows only; skipped when files are absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PI1M_PATH.exists(), reason="PI1M_v2.csv not downloaded")
class TestPI1MSmoke:
    def test_first_200_rows_have_high_keep_rate(self):
        rows = list(read_pi1m(PI1M_PATH, limit=200))
        assert len(rows) == 200
        assert all(r.count("*") == 2 for r in rows[:10])  # star-terminated PSMILES

        kept, stats = prepare_pselfies_corpus(rows)
        keep_rate = stats.n_kept / stats.n_input
        # Threshold justification: PI1M was generated by a model trained on
        # PoLyInfo polymers and already screened for RDKit validity, so parse
        # or terminus failures should be rare; the only expected attrition is
        # the occasional SELFIES-encoding failure or duplicate. 90% is a loose
        # floor -- in practice the measured rate is ~99%+.
        print(f"\nPI1M keep rate (first 200 rows): {keep_rate:.3f} ({stats.to_dict()})")
        assert keep_rate >= 0.90
        assert all(p.startswith("[") and p.endswith("]") for p in kept)


@pytest.mark.skipif(not LAMALAB_PATH.exists(), reason="LAMALAB_CURATED_Tg.csv not downloaded")
class TestLamaLabSmoke:
    def test_first_200_rows_parse_with_plausible_tg(self):
        pairs = list(read_lamalab_tg(LAMALAB_PATH, limit=200))
        assert len(pairs) == 200
        for psmiles, tg in pairs:
            assert isinstance(psmiles, str) and psmiles
            assert math.isfinite(tg)
            assert 80.0 <= tg <= 900.0  # plausible experimental Tg range in K

    def test_first_200_rows_convert_to_pselfies(self):
        pairs = list(read_lamalab_tg(LAMALAB_PATH, limit=200))
        kept, stats = prepare_labeled_corpus(pairs)
        keep_rate = stats.n_kept / stats.n_input
        print(f"\nLamaLab keep rate (first 200 rows): {keep_rate:.3f} ({stats.to_dict()})")
        # Curated, canonicalized PSMILES: expect nearly everything to convert.
        assert keep_rate >= 0.90


# ---------------------------------------------------------------------------
# Property value formatting (paper: "236.0", one decimal place)
# ---------------------------------------------------------------------------


class TestPropertyValueFormatting:
    def test_format_matches_paper(self):
        assert format_property_value(236.0) == "236.0"
        assert format_property_value(236.04) == "236.0"
        assert format_property_value(3.7) == "3.7"
        assert format_property_value(3.749, decimals=2) == "3.75"

    def test_parse_inverts_format(self):
        for x in (236.0, 3.7, -12.5, 0.0, 899.9):
            assert parse_property_value(format_property_value(x)) == pytest.approx(x)

    def test_parse_rejects_non_numeric(self):
        assert parse_property_value("soluble") is None
        assert parse_property_value("") is None
        assert parse_property_value("  ") is None
        assert parse_property_value("12.3.4") is None
        assert parse_property_value("nan") is None
        assert parse_property_value("inf") is None

    def test_parse_accepts_whitespace_padding(self):
        assert parse_property_value(" 236.0 ") == pytest.approx(236.0)


# ---------------------------------------------------------------------------
# Task example builders (paper SI formats, literal)
# ---------------------------------------------------------------------------


class TestExampleBuilders:
    def test_prediction_format_is_bare_pselfies_to_number(self):
        pairs = [("[At][C][C][O][At]", 236.04)]
        assert build_prediction_examples(pairs) == [("[At][C][C][O][At]", "236.0")]

    def test_prediction_has_no_task_prefix(self):
        (source, _target) = build_prediction_examples([("[At][C][C][At]", 100.0)])[0]
        assert source == "[At][C][C][At]"  # bare PSELFIES, nothing prepended

    def test_generation_format_is_number_to_pselfies(self):
        pairs = [("[At][C][C][O][At]", 236.04)]
        assert build_generation_examples(pairs) == [("236.0", "[At][C][C][O][At]")]

    def test_builders_are_inverses_of_each_other(self):
        pairs = [("[At][C][C][At]", 391.2), ("[At][C][O][At]", 200.0)]
        pred = build_prediction_examples(pairs)
        gen = build_generation_examples(pairs)
        assert [(t, s) for (s, t) in pred] == gen


# ---------------------------------------------------------------------------
# splits.py
# ---------------------------------------------------------------------------


class TestRandomSplit:
    def test_deterministic_per_seed(self):
        assert random_split(100, [0.9, 0.1], seed=7) == random_split(100, [0.9, 0.1], seed=7)

    def test_different_seeds_differ(self):
        assert random_split(100, [0.9, 0.1], seed=0) != random_split(100, [0.9, 0.1], seed=1)

    def test_exact_partition(self):
        parts = random_split(103, [0.8, 0.1, 0.1], seed=3)
        flat = sorted(i for part in parts for i in part)
        assert flat == list(range(103))

    def test_fractions_respected_within_one_element(self):
        n = 103
        fractions = [0.8, 0.1, 0.1]
        parts = random_split(n, fractions, seed=0)
        for part, frac in zip(parts, fractions, strict=True):
            assert abs(len(part) - frac * n) <= 1.0

    def test_bad_fractions_raise(self):
        with pytest.raises(ValueError):
            random_split(10, [0.5, 0.4], seed=0)  # sums to 0.9

    def test_pretraining_splits_shape(self):
        splits = make_pretraining_splits(1000, seed=0)
        assert set(splits) == {"train", "val", "test"}
        assert len(splits["train"]) == 900
        assert len(splits["val"]) == 50
        assert len(splits["test"]) == 50
        flat = sorted(splits["train"] + splits["val"] + splits["test"])
        assert flat == list(range(1000))


class TestKFoldRandomSplits:
    def test_five_folds_disjoint_and_complete(self):
        folds = make_kfold_random_splits(100, k=5, train_fraction=0.8, base_seed=0)
        assert len(folds) == 5
        for train_idx, test_idx in folds:
            assert not set(train_idx) & set(test_idx)
            assert sorted(train_idx + test_idx) == list(range(100))
            assert abs(len(train_idx) - 80) <= 1

    def test_folds_differ_from_each_other(self):
        folds = make_kfold_random_splits(200, k=5, base_seed=0)
        train_sets = [tuple(t) for t, _ in folds]
        assert len(set(train_sets)) == 5

    def test_different_base_seed_gives_different_folds(self):
        a = make_kfold_random_splits(100, k=5, base_seed=0)
        b = make_kfold_random_splits(100, k=5, base_seed=100)
        assert a != b

    def test_deterministic(self):
        assert make_kfold_random_splits(50, base_seed=3) == make_kfold_random_splits(
            50, base_seed=3
        )


class TestSplitsIO:
    def test_save_load_round_trip(self, tmp_path):
        splits = {
            "train": [0, 2, 4],
            "val": [1],
            "test": [3],
            "meta": {"seed": 0, "n": 5},
        }
        path = save_splits(tmp_path / "splits.json", splits)
        assert path.exists()
        assert load_splits(path) == splits


# ---------------------------------------------------------------------------
# datasets.py (fake tokenizer only; torch Datasets)
# ---------------------------------------------------------------------------


class TestDatasets:
    def _corpus(self):
        from polyt5.data.datasets import PSelfiesCorpus

        return PSelfiesCorpus(
            ["[At][C][C][O][At]", "[At][C][C][At]"], FakeTokenizer(), max_length=200
        )

    def test_pselfies_corpus_len_and_item_type(self):
        ds = self._corpus()
        assert len(ds) == 2
        item = ds[0]
        assert isinstance(item, list)
        assert all(isinstance(i, int) for i in item)
        assert item[-1] == FakeTokenizer.eos_id  # add_eos=True by default

    def test_pselfies_corpus_truncates(self):
        from polyt5.data.datasets import PSelfiesCorpus

        ds = PSelfiesCorpus(["[At]" + "[C]" * 100 + "[At]"], FakeTokenizer(), max_length=16)
        assert len(ds[0]) == 16

    def test_pselfies_corpus_stats_and_pickle(self):
        ds = self._corpus()
        assert ds.stats["n_examples"] == 2
        assert ds.stats["max_length"] == 200
        clone = pickle.loads(pickle.dumps(ds))  # num_workers>0 requirement
        assert clone[1] == ds[1]

    def test_seq2seq_dataset_returns_pair_of_id_lists(self):
        from polyt5.data.datasets import Seq2SeqDataset

        ds = Seq2SeqDataset(
            [("[At][C][C][O][At]", "236.0"), ("391.2", "[At][C][C][At]")],
            FakeTokenizer(),
            max_source_length=200,
            max_target_length=8,
        )
        assert len(ds) == 2
        src, tgt = ds[0]
        assert isinstance(src, list) and isinstance(tgt, list)
        assert all(isinstance(i, int) for i in src + tgt)
        assert tgt == FakeTokenizer().encode("236.0", add_eos=True)
        assert len(ds[1][1]) <= 8  # target truncation

    def test_seq2seq_dataset_stats_and_pickle(self):
        from polyt5.data.datasets import Seq2SeqDataset

        ds = Seq2SeqDataset([("a", "b")], FakeTokenizer())
        assert ds.stats["n_examples"] == 1
        clone = pickle.loads(pickle.dumps(ds))
        assert clone[0] == ds[0]


# ---------------------------------------------------------------------------
# sources.py (never touches the network)
# ---------------------------------------------------------------------------


class TestSources:
    def test_registry_contents(self):
        assert set(SOURCES) == {"pi1m", "lamalab_tg", "polyone_train", "polyone_dev"}
        for name, src in SOURCES.items():
            assert src.name == name
            assert src.url.startswith("https://")
            assert src.filename
            assert src.license
            assert src.citation
            assert src.approx_bytes > 0

    def test_polyone_requires_confirmation(self):
        assert SOURCES["polyone_train"].requires_confirmation
        assert SOURCES["polyone_dev"].requires_confirmation
        assert not SOURCES["pi1m"].requires_confirmation
        assert not SOURCES["lamalab_tg"].requires_confirmation

    def test_download_refuses_large_without_confirmation(self, tmp_path):
        with pytest.raises(ConfirmationRequiredError) as excinfo:
            download(SOURCES["polyone_train"], tmp_path)
        # The refusal must state the size so the user can decide.
        assert "GB" in str(excinfo.value) or "bytes" in str(excinfo.value)
        assert list(tmp_path.iterdir()) == []  # nothing fetched, nothing written

    def test_download_adopts_existing_file_without_network(self, tmp_path):
        # URL is unresolvable: any network attempt would raise, so a passing
        # test proves the existing file was adopted offline.
        src = DataSource(
            name="fake",
            url="https://invalid.invalid/fake.csv",
            filename="fake.csv",
            description="test fixture",
            license="MIT",
            approx_bytes=10,
            requires_confirmation=False,
            citation="none",
        )
        (tmp_path / "fake.csv").write_text("SMILES\n*CC*\n", encoding="utf-8")
        record = download(src, tmp_path)
        assert isinstance(record, DownloadRecord)
        assert record.name == "fake"
        assert record.bytes == (tmp_path / "fake.csv").stat().st_size
        assert len(record.sha256) == 64
        sidecar = tmp_path / "fake.csv.provenance.json"
        assert sidecar.exists()
        on_disk = json.loads(sidecar.read_text(encoding="utf-8"))
        assert on_disk["sha256"] == record.sha256
        assert on_disk["license"] == "MIT"

    def test_download_skip_is_stable_across_calls(self, tmp_path):
        src = replace(SOURCES["pi1m"], url="https://invalid.invalid/x.csv")
        (tmp_path / src.filename).write_text("SMILES,SA Score\n", encoding="utf-8")
        first = download(src, tmp_path)
        second = download(src, tmp_path)  # sidecar now exists: fast-path skip
        assert first.sha256 == second.sha256
        assert first.bytes == second.bytes

    def test_large_source_on_disk_needs_no_confirmation(self, tmp_path):
        # Confirmation gates network fetches, not reuse of what is already here.
        src = replace(SOURCES["polyone_dev"], url="https://invalid.invalid/y.txt")
        (tmp_path / src.filename).write_text("*CC*\n", encoding="utf-8")
        record = download(src, tmp_path)
        assert record.bytes > 0
