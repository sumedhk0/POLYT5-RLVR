"""Tests for the polyT5 span-corruption pretraining objective.

Ground truth (Sahu et al., npj Artificial Intelligence 2026):

    "The training objective follows the span corruption strategy introduced in the
    original T5 model. For each polymer sequence, up to 8 non-overlapping masked
    spans (each up to 3 tokens long) were randomly selected to mask up to 15% of
    the input tokens. These spans were replaced with sentinel tokens (<extra_id_n>)
    in the input sequence, and the target sequence was constructed by concatenating
    the masked spans, each prefixed with its corresponding sentinel token. The
    sentinel tokens were assigned in increasing numerical order of n and placed
    such that no two masked spans were adjacent, ensuring at least one unmasked
    token between them."

The strongest check in this file is `reconstruct`: given only (input_ids, labels,
sentinel_ids) the original sequence must be rebuildable byte-for-byte.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polyt5.data.span_corruption import (  # noqa: E402
    DEFAULT_CONFIG,
    SpanCorruptionConfig,
    SpanCorruptionResult,
    batch_span_corrupt,
    corruption_statistics,
    span_corrupt,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

# 33 sentinels: enough for 8 spans + final sentinel with a wide margin, and ids
# far away from ordinary token ids so accidental collisions are impossible.
SENTINELS = list(range(1000, 1033))
EOS = 1
PAD = 0


def random_sequence(rng: np.random.Generator, length: int) -> list[int]:
    """Random token ids in [5, 500), disjoint from SENTINELS / EOS / PAD."""
    return [int(x) for x in rng.integers(5, 500, size=length)]


def reconstruct(
    input_ids: list[int], labels: list[int], sentinel_ids: list[int]
) -> list[int]:
    """Rebuild the original sequence from a corrupted (input, target) pair.

    Parses ``labels`` into {sentinel: masked span tokens}, then walks
    ``input_ids`` replacing each sentinel with its span. The trailing final
    sentinel in ``labels`` (and anything after it, e.g. an appended EOS) never
    appears in ``input_ids``, so it is harmlessly ignored.
    """
    sentinel_set = set(sentinel_ids)
    span_map: dict[int, list[int]] = {}
    current: int | None = None
    for tok in labels:
        if tok in sentinel_set:
            current = tok
            span_map[current] = []
        else:
            assert current is not None, "labels must start with a sentinel"
            span_map[current].append(tok)
    out: list[int] = []
    for tok in input_ids:
        if tok in sentinel_set:
            out.extend(span_map[tok])
        else:
            out.append(tok)
    return out


def spans_are_valid(result: SpanCorruptionResult, seq_len: int, min_gap: int) -> None:
    """Assert the paper's structural constraints on a single result."""
    spans = result.spans
    # ascending, in-bounds, positive lengths
    for start, length in spans:
        assert length >= 1
        assert start >= 0
        assert start + length <= seq_len, "span runs past the sequence end"
    for (s0, l0), (s1, _l1) in zip(spans, spans[1:], strict=False):
        # NON-ADJACENCY: >= min_gap unmasked tokens between consecutive spans.
        assert s1 >= s0 + l0 + min_gap, f"adjacent/overlapping spans: {spans}"


# ---------------------------------------------------------------------------
# Corruption-rate statistics
# ---------------------------------------------------------------------------


class TestCorruptionFraction:
    def test_statistics_over_500_sequences(self):
        rng = np.random.default_rng(2026)
        n_seq, length = 500, 200
        fractions = []
        for _ in range(n_seq):
            seq = random_sequence(rng, length)
            res = span_corrupt(seq, sentinel_ids=SENTINELS, rng=rng)
            assert res.n_masked > 0, "length-200 sequences must always be maskable"
            fractions.append(res.corruption_fraction)
            # HARD invariant from the paper: never mask more than 15%.
            assert res.corruption_fraction <= 0.15 + 1e-12
            assert res.n_masked <= 0.15 * length + 1e-9
        mean = float(np.mean(fractions))
        # Tolerance justification: with the paper's hard caps (<=8 spans, each
        # uniform on {1,2,3} tokens) the masked count per length-200 sequence is
        # a sum of 8 iid U{1,2,3} draws -> mean 16 tokens = 8.0% (the 15% cap
        # allows 30 tokens, the 8x3 cap allows 24, so neither truncates the
        # draw). Per-sequence sd = sqrt(8 * 2/3) ~= 2.31 tokens = 1.15%, so the
        # standard error of the mean over 500 sequences is ~0.052%. A +-1%
        # window around the analytic 8% expectation is ~19 standard errors:
        # essentially impossible to fail by chance, tight enough to catch any
        # systematic bias in length sampling or budget accounting.
        assert mean <= 0.15
        assert abs(mean - 0.08) < 0.01, f"mean fraction {mean:.4f} far from analytic 0.08"

    def test_fraction_definition(self):
        rng = np.random.default_rng(7)
        seq = random_sequence(rng, 200)
        res = span_corrupt(seq, sentinel_ids=SENTINELS, rng=rng)
        assert res.corruption_fraction == pytest.approx(res.n_masked / len(seq))


# ---------------------------------------------------------------------------
# Structural caps and the paper's non-adjacency constraint
# ---------------------------------------------------------------------------


class TestSpanStructure:
    @pytest.mark.parametrize("length", [8, 20, 50, 161, 200, 331])
    def test_caps_and_non_adjacency_many_seeds(self, length):
        for seed in range(200):
            rng = np.random.default_rng(seed)
            seq = random_sequence(rng, length)
            res = span_corrupt(seq, sentinel_ids=SENTINELS, rng=rng)
            assert len(res.spans) <= DEFAULT_CONFIG.max_spans
            assert all(ln <= DEFAULT_CONFIG.max_span_length for _, ln in res.spans)
            spans_are_valid(res, length, DEFAULT_CONFIG.min_gap)
            assert res.n_masked == sum(ln for _, ln in res.spans)

    def test_spans_sorted_ascending(self):
        rng = np.random.default_rng(11)
        for _ in range(50):
            seq = random_sequence(rng, 120)
            res = span_corrupt(seq, sentinel_ids=SENTINELS, rng=rng)
            starts = [s for s, _ in res.spans]
            assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# Sentinel ordering
# ---------------------------------------------------------------------------


class TestSentinelOrdering:
    def test_input_and_label_sentinels_increasing(self):
        rng = np.random.default_rng(23)
        sentinel_set = set(SENTINELS)
        for _ in range(100):
            seq = random_sequence(rng, 150)
            res = span_corrupt(seq, sentinel_ids=SENTINELS, rng=rng)
            n = len(res.spans)
            in_sent = [t for t in res.input_ids if t in sentinel_set]
            lab_sent = [t for t in res.labels if t in sentinel_set]
            # exactly sentinel_ids[0..n-1], strictly increasing left to right
            assert in_sent == SENTINELS[:n]
            # labels additionally carry the trailing final sentinel
            assert lab_sent == SENTINELS[: n + 1]


# ---------------------------------------------------------------------------
# Reconstruction round-trip (the strongest correctness check)
# ---------------------------------------------------------------------------


class TestReconstruction:
    def test_roundtrip_hundreds_of_cases(self):
        rng = np.random.default_rng(99)
        for case in range(400):
            length = int(rng.integers(0, 300))
            seq = random_sequence(rng, length)
            res = span_corrupt(seq, sentinel_ids=SENTINELS, rng=rng)
            if res.n_masked == 0:
                assert res.input_ids == seq
                assert res.labels == []
            else:
                rebuilt = reconstruct(res.input_ids, res.labels, SENTINELS)
                assert rebuilt == seq, f"round-trip failed at case {case} (len={length})"

    def test_roundtrip_with_eos(self):
        rng = np.random.default_rng(101)
        for _ in range(100):
            seq = random_sequence(rng, 180) + [EOS]
            res = span_corrupt(seq, sentinel_ids=SENTINELS, rng=rng, eos_id=EOS)
            assert reconstruct(res.input_ids, res.labels, SENTINELS) == seq


# ---------------------------------------------------------------------------
# Target format
# ---------------------------------------------------------------------------


class TestTargetFormat:
    def test_layout_without_eos(self):
        rng = np.random.default_rng(31)
        seq = random_sequence(rng, 200)
        res = span_corrupt(seq, sentinel_ids=SENTINELS, rng=rng)
        n = len(res.spans)
        assert res.labels[0] == SENTINELS[0]
        assert res.labels[-1] == SENTINELS[n]  # trailing final sentinel
        # labels are exactly: [S_i, span_i tokens]* then S_n
        expected: list[int] = []
        for i, (start, ln) in enumerate(res.spans):
            expected.append(SENTINELS[i])
            expected.extend(seq[start : start + ln])
        expected.append(SENTINELS[n])
        assert res.labels == expected

    def test_layout_with_eos_appended(self):
        # Chosen rule (documented in span_corrupt): when eos_id is given, it is
        # appended to labels AFTER the final sentinel, mirroring HF T5.
        rng = np.random.default_rng(37)
        seq = random_sequence(rng, 200) + [EOS]
        res = span_corrupt(seq, sentinel_ids=SENTINELS, rng=rng, eos_id=EOS)
        n = len(res.spans)
        assert res.labels[-1] == EOS
        assert res.labels[-2] == SENTINELS[n]

    def test_no_final_sentinel_config(self):
        cfg = SpanCorruptionConfig(add_final_sentinel=False)
        rng = np.random.default_rng(41)
        seq = random_sequence(rng, 200)
        res = span_corrupt(seq, sentinel_ids=SENTINELS, config=cfg, rng=rng)
        n = len(res.spans)
        sentinel_set = set(SENTINELS)
        assert [t for t in res.labels if t in sentinel_set] == SENTINELS[:n]
        last_start, last_len = res.spans[-1]
        assert res.labels[-last_len:] == seq[last_start : last_start + last_len]


# ---------------------------------------------------------------------------
# EOS handling
# ---------------------------------------------------------------------------


class TestEosHandling:
    def test_trailing_eos_never_masked(self):
        for seed in range(100):
            rng = np.random.default_rng(seed)
            seq = random_sequence(rng, 60) + [EOS]
            res = span_corrupt(seq, sentinel_ids=SENTINELS, rng=rng, eos_id=EOS)
            last = len(seq) - 1
            for start, ln in res.spans:
                assert start + ln <= last, "span covers the trailing EOS"
            assert res.input_ids[-1] == EOS

    def test_eos_id_none_allows_masking_last_token(self):
        # With no eos_id declared there is no protected suffix; over many seeds
        # the final position must eventually be masked, proving the exclusion
        # is driven by eos_id and not an off-by-one in placement.
        hit_last = False
        for seed in range(200):
            rng = np.random.default_rng(seed)
            seq = random_sequence(rng, 60)
            res = span_corrupt(seq, sentinel_ids=SENTINELS, rng=rng)
            if any(start + ln == len(seq) for start, ln in res.spans):
                hit_last = True
                break
        assert hit_last


# ---------------------------------------------------------------------------
# Degenerate / short inputs
# ---------------------------------------------------------------------------


class TestShortSequences:
    @pytest.mark.parametrize("length", [0, 1, 2, 3])
    def test_too_short_returns_unchanged(self, length):
        rng = np.random.default_rng(5)
        seq = random_sequence(rng, length)
        res = span_corrupt(seq, sentinel_ids=SENTINELS, rng=rng)
        assert res.input_ids == seq
        assert res.labels == []
        assert res.spans == ()
        assert res.n_masked == 0
        assert res.corruption_fraction == 0.0

    def test_eos_only_sequence(self):
        rng = np.random.default_rng(6)
        res = span_corrupt([EOS], sentinel_ids=SENTINELS, rng=rng, eos_id=EOS)
        assert res.input_ids == [EOS]
        assert res.n_masked == 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_identical(self):
        seq_rng = np.random.default_rng(0)
        seqs = [random_sequence(seq_rng, 200) for _ in range(20)]
        res_a = [
            span_corrupt(s, sentinel_ids=SENTINELS, rng=np.random.default_rng(1234))
            for s in seqs
        ]
        res_b = [
            span_corrupt(s, sentinel_ids=SENTINELS, rng=np.random.default_rng(1234))
            for s in seqs
        ]
        assert res_a == res_b  # dataclass equality: byte-for-byte identical fields

    def test_different_seeds_differ(self):
        seq = random_sequence(np.random.default_rng(0), 200)
        res_a = span_corrupt(seq, sentinel_ids=SENTINELS, rng=np.random.default_rng(1234))
        res_b = span_corrupt(seq, sentinel_ids=SENTINELS, rng=np.random.default_rng(4321))
        assert res_a != res_b


# ---------------------------------------------------------------------------
# Paper Figure 2C reproduction
# ---------------------------------------------------------------------------


class TestFigure2C:
    def test_exact_layout(self):
        # Token-id aliases for the figure's PSELFIES tokens.
        C, BR1, AT, EQC, BR2, RING1, OX, EQBR1 = 10, 11, 12, 13, 14, 15, 16, 17
        original = [
            C, C, BR1, C, C, BR1, C, AT, C, EQC, C, EQC, BR2, RING1, C, OX, C, EQBR1
        ]
        s0, s1, s2, s3 = SENTINELS[:4]
        rng = np.random.default_rng(0)  # unused on the forced path, still required
        res = span_corrupt(
            original,
            sentinel_ids=SENTINELS,
            rng=rng,
            _forced_spans=((1, 3), (7, 3), (12, 2)),
        )
        # corrupted input from the figure:
        # [C] <e0> [C] [Branch1] [C] <e1> [C] [=C] <e2> [C] [O] [C] [=Branch1]
        assert res.input_ids == [C, s0, C, BR1, C, s1, C, EQC, s2, C, OX, C, EQBR1]
        # target from the figure (with the trailing final sentinel):
        # <e0> [C] [Branch1] [C] <e1> [At] [C] [=C] <e2> [Branch2] [Ring1] <e3>
        assert res.labels == [s0, C, BR1, C, s1, AT, C, EQC, s2, BR2, RING1, s3]
        assert res.spans == ((1, 3), (7, 3), (12, 2))
        assert res.n_masked == 8
        # gaps in the figure: 3 unmasked tokens after span 0, 2 after span 1
        assert reconstruct(res.input_ids, res.labels, SENTINELS) == original


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_spans": 0},
            {"max_span_length": 0},
            {"corruption_rate": 0.0},
            {"corruption_rate": 1.5},
            {"min_gap": -1},
        ],
    )
    def test_invalid_config_raises(self, kwargs):
        with pytest.raises(ValueError):
            SpanCorruptionConfig(**kwargs)

    def test_default_config_matches_paper(self):
        assert DEFAULT_CONFIG.max_spans == 8
        assert DEFAULT_CONFIG.max_span_length == 3
        assert DEFAULT_CONFIG.corruption_rate == 0.15
        assert DEFAULT_CONFIG.min_gap == 1
        assert DEFAULT_CONFIG.add_final_sentinel is True
        assert DEFAULT_CONFIG.mask_eos is False


# ---------------------------------------------------------------------------
# batch_span_corrupt / corruption_statistics
# ---------------------------------------------------------------------------


class TestBatchAndStatistics:
    def test_batch_deterministic_and_valid(self):
        seq_rng = np.random.default_rng(0)
        seqs = [random_sequence(seq_rng, int(seq_rng.integers(0, 250))) for _ in range(50)]
        out_a = batch_span_corrupt(
            seqs, sentinel_ids=SENTINELS, rng=np.random.default_rng(9)
        )
        out_b = batch_span_corrupt(
            seqs, sentinel_ids=SENTINELS, rng=np.random.default_rng(9)
        )
        assert out_a == out_b
        assert len(out_a) == len(seqs)
        for seq, res in zip(seqs, out_a, strict=True):
            if res.n_masked:
                assert reconstruct(res.input_ids, res.labels, SENTINELS) == seq

    def test_statistics_keys_and_values(self):
        rng = np.random.default_rng(3)
        seqs = [random_sequence(rng, 200) for _ in range(50)]
        seqs += [random_sequence(rng, 2) for _ in range(50)]  # guaranteed skips
        results = batch_span_corrupt(seqs, sentinel_ids=SENTINELS, rng=rng)
        stats = corruption_statistics(results)
        for key in (
            "mean_corruption_fraction",
            "median_corruption_fraction",
            "mean_span_count",
            "mean_span_length",
            "fraction_skipped",
        ):
            assert key in stats
            assert isinstance(stats[key], float)
        assert stats["fraction_skipped"] == pytest.approx(0.5)
        assert 0.0 < stats["mean_corruption_fraction"] <= 0.15
        assert 0.0 < stats["mean_span_count"] <= 8.0
        assert 1.0 <= stats["mean_span_length"] <= 3.0

    def test_statistics_empty_input(self):
        stats = corruption_statistics([])
        assert stats["fraction_skipped"] == 0.0
        assert stats["mean_corruption_fraction"] == 0.0


# ---------------------------------------------------------------------------
# pad_sequences (pure python -- no torch needed)
# ---------------------------------------------------------------------------


class TestPadSequences:
    def test_basic_padding_and_mask(self):
        from polyt5.data.collate import pad_sequences

        padded, mask = pad_sequences([[1, 2, 3], [4], []], pad_id=PAD)
        assert padded == [[1, 2, 3], [4, PAD, PAD], [PAD, PAD, PAD]]
        assert mask == [[1, 1, 1], [1, 0, 0], [0, 0, 0]]

    def test_truncation_to_max_length(self):
        from polyt5.data.collate import pad_sequences

        padded, mask = pad_sequences([[1, 2, 3, 4, 5], [6]], pad_id=PAD, max_length=3)
        assert padded == [[1, 2, 3], [6, PAD, PAD]]
        assert mask == [[1, 1, 1], [1, 0, 0]]

    def test_pad_to_multiple_of(self):
        from polyt5.data.collate import pad_sequences

        padded, mask = pad_sequences([[1, 2, 3], [4]], pad_id=PAD, pad_to_multiple_of=8)
        assert all(len(row) == 8 for row in padded)
        assert all(len(row) == 8 for row in mask)


# ---------------------------------------------------------------------------
# Torch collators (skipped until torch finishes installing)
# ---------------------------------------------------------------------------


class TestSpanCorruptionCollator:
    def _make(self, seed=0, **kw):
        from polyt5.data.collate import SpanCorruptionCollator

        return SpanCorruptionCollator(
            sentinel_ids=SENTINELS, pad_id=PAD, eos_id=EOS, seed=seed, **kw
        )

    def test_shapes_masks_and_label_padding(self):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(0)
        batch = [random_sequence(rng, n) + [EOS] for n in (50, 120, 199)]
        out = self._make()(batch)
        assert set(out) == {"input_ids", "attention_mask", "labels"}
        for t in out.values():
            assert t.dtype == torch.long
            assert t.shape[0] == 3
        assert out["input_ids"].shape == out["attention_mask"].shape
        # attention_mask is 1 exactly on non-pad positions
        assert torch.equal(out["attention_mask"], (out["input_ids"] != PAD).long())
        # label padding uses -100, never pad_id: every row's tail is -100 and
        # -100 appears in place of any pad token
        labels = out["labels"]
        row_lens = (labels != -100).sum(dim=1)
        for i in range(labels.shape[0]):
            assert (labels[i, row_lens[i] :] == -100).all()
            assert (labels[i, : row_lens[i]] != -100).all()

    def test_truncation_to_max_length(self):
        pytest.importorskip("torch")
        rng = np.random.default_rng(1)
        batch = [random_sequence(rng, 400)]
        out = self._make(max_length=200)(batch)
        # corruption shortens the truncated input (spans collapse to sentinels)
        assert out["input_ids"].shape[1] <= 200

    def test_deterministic_with_seed(self):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(2)
        batch = [random_sequence(rng, 150) for _ in range(4)]
        out_a = self._make(seed=77)(batch)
        out_b = self._make(seed=77)(batch)
        for key in out_a:
            assert torch.equal(out_a[key], out_b[key])

    def test_set_epoch_changes_masking(self):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(3)
        batch = [random_sequence(rng, 200) for _ in range(4)]
        coll = self._make(seed=77)
        out_e0 = coll(batch)
        coll.set_epoch(1)
        out_e1 = coll(batch)
        assert not (
            out_e0["input_ids"].shape == out_e1["input_ids"].shape
            and torch.equal(out_e0["input_ids"], out_e1["input_ids"])
        )
        # ...and set_epoch(0) restores epoch-0 masking (resume reproducibility)
        coll.set_epoch(0)
        out_e0_again = coll(batch)
        for key in out_e0:
            assert torch.equal(out_e0[key], out_e0_again[key])

    def test_no_seed_differs_between_calls(self):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(4)
        batch = [random_sequence(rng, 200) for _ in range(4)]
        coll = self._make(seed=None)
        out_a = coll(batch)
        out_b = coll(batch)
        assert not (
            out_a["input_ids"].shape == out_b["input_ids"].shape
            and torch.equal(out_a["input_ids"], out_b["input_ids"])
        )


class TestSeq2SeqCollator:
    def test_no_masking_and_label_padding(self):
        torch = pytest.importorskip("torch")
        from polyt5.data.collate import Seq2SeqCollator

        coll = Seq2SeqCollator(pad_id=PAD)
        batch = [([5, 6, 7, EOS], [8, 9, EOS]), ([10, EOS], [11, 12, 13, 14, EOS])]
        out = coll(batch)
        assert set(out) == {"input_ids", "attention_mask", "labels"}
        # sources pass through unmodified (no sentinels, no corruption)
        assert out["input_ids"][0].tolist() == [5, 6, 7, EOS]
        assert out["input_ids"][1].tolist() == [10, EOS, PAD, PAD]
        assert out["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
        assert out["labels"][0].tolist() == [8, 9, EOS, -100, -100]
        assert out["labels"][1].tolist() == [11, 12, 13, 14, EOS]
        assert out["labels"].dtype == torch.long

    def test_truncation(self):
        pytest.importorskip("torch")
        from polyt5.data.collate import Seq2SeqCollator

        coll = Seq2SeqCollator(pad_id=PAD, max_source_length=3, max_target_length=2)
        out = coll([(list(range(10)), list(range(10)))])
        assert out["input_ids"].shape[1] == 3
        assert out["labels"].shape[1] == 2
