"""Tests for the inference/predictor layer that closes the evaluation loop.

CPU-only and fast. Two model fixtures are used:

* an **untrained** tiny polyT5, which decodes garbage -- this is the adversarial
  case the predictor must survive without raising;
* a **briefly trained** tiny polyT5 (a few hundred steps on a constant target),
  which actually emits ``"500.0"``. It exists so the injection into
  :func:`polyt5.evaluation.evaluate_generation` can be exercised for real,
  producing non-``None`` property statistics rather than the "no numbers came
  back" degenerate path.

Nothing here trains a paper-sized model or touches the real Tg corpus.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from polyt5.evaluation import evaluate_generation, target_property_rate
from polyt5.inference import (
    NON_NUMERIC_VALUE,
    CachedPredictor,
    EnsemblePropertyPredictor,
    PolyT5PropertyPredictor,
    PredictionResult,
    conditioning_report,
)
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.tokenization import PolyT5Tokenizer
from polyt5.training import save_checkpoint
from polyt5.utils import seed_everything

REPO = Path(__file__).resolve().parents[1]
RUN_SPLITS_PATH = REPO / "scripts" / "run_splits.py"
EVALUATE_GENERATIONS_PATH = REPO / "scripts" / "evaluate_generations.py"

#: A handful of PSELFIES that survive the whole SV/TSD/DD/PV cascade.
VALID_PSELFIES = [
    "[At][C][C][O][At]",
    "[At][C][C][C][At]",
    "[At][C][O][C][C][At]",
    "[At][C][C][C][C][At]",
]

TRAINED_TARGET = "500.0"


# --------------------------------------------------------------------- helpers
def _tiny_config(tokenizer: PolyT5Tokenizer) -> PolyT5Config:
    """A deliberately tiny model config wired to the real tokenizer's ids."""
    return PolyT5Config(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        d_kv=16,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        n_positions=64,
        dropout_rate=0.0,
        pad_token_id=tokenizer.pad_id,
        eos_token_id=tokenizer.eos_id,
        decoder_start_token_id=tokenizer.decoder_start_token_id,
    )


def _write_checkpoint(
    directory: Path,
    model: PolyT5ForConditionalGeneration,
    tokenizer: PolyT5Tokenizer,
    *,
    tokenizer_sha256: str | None = None,
) -> tuple[Path, Path]:
    """Save ``model`` plus a tokenizer artifact; return ``(ckpt, tokenizer)``."""
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer_path = directory / "vocab.json"
    tokenizer.save(tokenizer_path)
    checkpoint_path = directory / "best.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        epoch=0,
        global_step=0,
        config={"task": "tg_prediction"},
        model_config=model.config.to_dict(),
        tokenizer_path=tokenizer_path,
        tokenizer_sha256=tokenizer_sha256 or tokenizer.sha256,
    )
    return checkpoint_path, tokenizer_path


# -------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def tokenizer() -> PolyT5Tokenizer:
    return PolyT5Tokenizer.default()


@pytest.fixture(scope="module")
def untrained_checkpoint(tmp_path_factory, tokenizer) -> Path:
    """A checkpoint holding a randomly initialised tiny model."""
    seed_everything(0)
    model = PolyT5ForConditionalGeneration(_tiny_config(tokenizer))
    directory = tmp_path_factory.mktemp("untrained")
    checkpoint_path, _ = _write_checkpoint(directory, model, tokenizer)
    return checkpoint_path


@pytest.fixture(scope="module")
def mismatched_checkpoint(tmp_path_factory, tokenizer) -> Path:
    """A checkpoint whose recorded tokenizer hash belongs to another vocabulary."""
    seed_everything(0)
    model = PolyT5ForConditionalGeneration(_tiny_config(tokenizer))
    directory = tmp_path_factory.mktemp("mismatched")
    checkpoint_path, _ = _write_checkpoint(
        directory, model, tokenizer, tokenizer_sha256="0" * 64
    )
    return checkpoint_path


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory, tokenizer) -> Path:
    """A tiny model briefly trained to emit a constant numeric target.

    Memorising an unconditional decoder sequence is easy enough that a couple of
    hundred CPU steps suffice; the assertion in
    :func:`test_trained_predictor_emits_numbers` fails loudly if it ever stops
    being enough.
    """
    seed_everything(0)
    model = PolyT5ForConditionalGeneration(_tiny_config(tokenizer))
    encoded = tokenizer.batch_encode(VALID_PSELFIES, add_eos=True, max_length=64)
    input_ids = torch.tensor(encoded["input_ids"])
    attention_mask = torch.tensor(encoded["attention_mask"])
    label_rows = tokenizer.batch_encode(
        [TRAINED_TARGET] * len(VALID_PSELFIES), add_eos=True, max_length=16
    )
    labels = torch.tensor(label_rows["input_ids"])
    labels[labels == tokenizer.pad_id] = -100

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(200):
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    directory = tmp_path_factory.mktemp("trained")
    checkpoint_path, _ = _write_checkpoint(directory, model, tokenizer)
    return checkpoint_path


@pytest.fixture(scope="module")
def untrained_predictor(untrained_checkpoint) -> PolyT5PropertyPredictor:
    return PolyT5PropertyPredictor.from_checkpoint(untrained_checkpoint, device="cpu")


@pytest.fixture(scope="module")
def trained_predictor(trained_checkpoint) -> PolyT5PropertyPredictor:
    return PolyT5PropertyPredictor.from_checkpoint(
        trained_checkpoint, device="cpu", property_name="Tg"
    )


# ------------------------------------------------------------------- loading
def test_from_checkpoint_loads_and_predicts(untrained_predictor):
    """A saved tiny checkpoint loads and scores without raising."""
    results = untrained_predictor.predict(VALID_PSELFIES)
    assert len(results) == len(VALID_PSELFIES)
    assert all(isinstance(result, PredictionResult) for result in results)


def test_from_checkpoint_uses_tokenizer_recorded_in_checkpoint(untrained_checkpoint, tokenizer):
    """With no explicit tokenizer, the one named in the checkpoint is loaded."""
    predictor = PolyT5PropertyPredictor.from_checkpoint(untrained_checkpoint, device="cpu")
    assert predictor.tokenizer.sha256 == tokenizer.sha256


def test_tokenizer_sha_mismatch_raises_value_error(mismatched_checkpoint):
    """Scoring across vocabularies would silently corrupt every token id."""
    with pytest.raises(ValueError) as excinfo:
        PolyT5PropertyPredictor.from_checkpoint(mismatched_checkpoint, device="cpu")
    message = str(excinfo.value)
    assert "tokenizer" in message.lower()
    assert "0000000000000000" in message  # the checkpoint's (bogus) hash


def test_missing_tokenizer_path_raises_clear_error(tmp_path, tokenizer):
    """A checkpoint with no recorded tokenizer must say so, not guess."""
    seed_everything(0)
    model = PolyT5ForConditionalGeneration(_tiny_config(tokenizer))
    checkpoint_path = tmp_path / "no_tokenizer.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        epoch=0,
        global_step=0,
        config={},
        model_config=model.config.to_dict(),
        tokenizer_path=None,
        tokenizer_sha256=tokenizer.sha256,
    )
    with pytest.raises(ValueError, match="tokenizer_path"):
        PolyT5PropertyPredictor.from_checkpoint(checkpoint_path, device="cpu")


# ------------------------------------------------------------------ predicting
def test_predict_returns_one_result_per_input_in_order(untrained_predictor):
    """One :class:`PredictionResult` per input, positionally aligned."""
    sources = VALID_PSELFIES + ["", "not a polymer at all"]
    results = untrained_predictor.predict(sources)
    assert len(results) == len(sources)
    assert [result.source for result in results] == sources


def test_untrained_model_is_non_numeric_and_never_raises(untrained_predictor):
    """Garbage decodes are recorded as failures, never exceptions."""
    results = untrained_predictor.predict(VALID_PSELFIES)
    assert any(not result.is_numeric for result in results)
    for result in results:
        if not result.is_numeric:
            assert result.value is None

    values = untrained_predictor.predict_values(VALID_PSELFIES)
    assert all(value is None for value, result in zip(values, results, strict=True)
               if not result.is_numeric)

    floats = untrained_predictor(VALID_PSELFIES)
    assert len(floats) == len(VALID_PSELFIES)
    for value, result in zip(floats, results, strict=True):
        if not result.is_numeric:
            assert math.isnan(value)


def test_predictions_are_deterministic(untrained_predictor):
    """Beam search over a fixed model is a pure function of its input."""
    first = untrained_predictor.predict(VALID_PSELFIES)
    second = untrained_predictor.predict(VALID_PSELFIES)
    assert [r.decoded for r in first] == [r.decoded for r in second]
    assert [r.value for r in first] == [r.value for r in second]


def test_batching_does_not_change_predictions(trained_checkpoint):
    """A prediction must not depend on which batch its input landed in."""
    single = PolyT5PropertyPredictor.from_checkpoint(
        trained_checkpoint, device="cpu", batch_size=1
    )
    batched = PolyT5PropertyPredictor.from_checkpoint(
        trained_checkpoint, device="cpu", batch_size=64
    )
    assert single.predict_values(VALID_PSELFIES) == batched.predict_values(VALID_PSELFIES)


def test_trained_predictor_emits_numbers(trained_predictor):
    """The trained fixture really does decode to a parseable number."""
    results = trained_predictor.predict(VALID_PSELFIES)
    assert all(result.is_numeric for result in results), [r.decoded for r in results]
    assert all(result.value == pytest.approx(500.0) for result in results)


def test_predict_psmiles_convenience(trained_predictor):
    """PSMILES input is accepted through the documented convenience method."""
    psmiles = ["[*]CCO[*]", "[At]CCC[At]"]
    results = trained_predictor.predict_psmiles(psmiles)
    assert [result.source for result in results] == psmiles
    assert all(result.is_numeric for result in results)


def test_call_accepts_psmiles_and_pselfies(trained_predictor):
    """The injection point auto-detects both notations (evaluate_generation
    hands it canonical PSMILES)."""
    values = trained_predictor(["[At][C][C][O][At]", "[At]CCO[At]", "[*]CCO[*]"])
    assert all(value == pytest.approx(500.0) for value in values)


def test_call_returns_nan_for_unusable_input(trained_predictor):
    """Documented contract: a non-numeric / unconvertible entry maps to NaN."""
    values = trained_predictor(["", "definitely not chemistry {}[]"])
    assert len(values) == 2
    assert all(math.isnan(value) for value in values)
    assert math.isnan(NON_NUMERIC_VALUE)


def test_call_is_a_sequence_to_floats_callable(trained_predictor):
    """``Callable[[Sequence[str]], Sequence[float]]``, including empty input."""
    injected: object = trained_predictor
    assert callable(injected)
    assert list(trained_predictor([])) == []
    values = trained_predictor(tuple(VALID_PSELFIES))
    assert len(values) == len(VALID_PSELFIES)
    assert all(isinstance(value, float) for value in values)


# ------------------------------------------------- wiring into the evaluation
def test_predictor_wires_into_evaluate_generation(trained_predictor):
    """The real end-to-end loop: TP is computed instead of coming back None."""
    report = evaluate_generation(
        VALID_PSELFIES,
        target_property=500.0,
        tolerance=50.0,
        property_predictor=trained_predictor,
        compute_sa=False,
    )
    assert report.n_property_values is not None and report.n_property_values > 0
    assert report.property_mean is not None
    assert report.property_median is not None
    assert report.property_std is not None
    assert report.property_target == 500.0
    assert report.property_tolerance == 50.0
    assert report.target_property_rate is not None
    assert 0.0 <= report.target_property_rate <= 1.0
    assert report.target_property_rate == pytest.approx(1.0)
    assert report.to_dict()["target_property_rate"] == pytest.approx(1.0)


def test_non_finite_predictions_are_excluded_from_tp():
    """NaN means "no answer", not "missed the window"; it leaves the denominator.

    Hand-computed: finite values are 490, 510 and 600 K. Two of the three lie
    inside 500 +- 50 K, so TP is 2/3 -- NOT 2/5, which is what counting the two
    NaNs as failures would give.
    """
    mixed = [490.0, 510.0, 600.0, float("nan"), float("nan")]
    assert target_property_rate(mixed, 500.0, 50.0) == pytest.approx(2.0 / 3.0)

    def fake_predictor(candidates: Sequence[str]) -> list[float]:
        del candidates
        return mixed

    report = evaluate_generation(
        VALID_PSELFIES,
        target_property=500.0,
        tolerance=50.0,
        property_predictor=fake_predictor,
        compute_sa=False,
    )
    assert report.n_property_values == 3
    assert report.target_property_rate == pytest.approx(2.0 / 3.0)


# ------------------------------------------------------------- CachedPredictor
class _CountingPredictor:
    """A predictor that records exactly which candidates it was asked to score."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, candidates: Sequence[str]) -> list[float]:
        self.calls.append(list(candidates))
        return [float(len(candidate)) for candidate in candidates]

    @property
    def n_scored(self) -> int:
        return sum(len(call) for call in self.calls)


def test_cached_predictor_matches_and_avoids_recomputation():
    inner = _CountingPredictor()
    cached = CachedPredictor(inner)

    candidates = ["[At][C][C][O][At]", "[At][C][C][At]", "[At][C][C][O][At]"]
    first = list(cached(candidates))
    assert first == list(_CountingPredictor()(candidates))
    # The repeated candidate is scored once, not twice.
    assert inner.n_scored == 2

    second = list(cached(candidates))
    assert second == first
    assert inner.n_scored == 2  # nothing recomputed

    # Counters are per LOOKUP, as in functools.lru_cache: 2 calls x 3
    # candidates = 6 lookups. The first call's duplicate is a miss (nothing was
    # cached yet when it was looked up) but is deduplicated before the wrapped
    # predictor is invoked, which is what `inner.n_scored == 2` above proves.
    info = cached.cache_info()
    assert info.hits + info.misses == 6
    assert info.misses == 3
    assert info.hits == 3
    assert info.currsize == 2

    cached.clear()
    assert cached.cache_info().currsize == 0


def test_cached_predictor_evicts_beyond_maxsize():
    inner = _CountingPredictor()
    cached = CachedPredictor(inner, maxsize=2)
    cached(["a", "b", "c"])
    assert cached.cache_info().currsize == 2
    assert cached.cache_info().maxsize == 2


def test_cached_predictor_wraps_the_real_predictor(trained_predictor):
    cached = CachedPredictor(trained_predictor)
    assert list(cached(VALID_PSELFIES)) == list(trained_predictor(VALID_PSELFIES))
    assert cached.cache_info().currsize == len(VALID_PSELFIES)


# -------------------------------------------------- EnsemblePropertyPredictor
class _ConstantPredictor:
    """A member that always returns the same value for every candidate."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, candidates: Sequence[str]) -> list[float]:
        return [self.value] * len(candidates)


def test_ensemble_mean_is_the_mean_of_its_members():
    ensemble = EnsemblePropertyPredictor(
        [_ConstantPredictor(400.0), _ConstantPredictor(500.0), _ConstantPredictor(600.0)]
    )
    assert len(ensemble) == 3
    values = ensemble(VALID_PSELFIES)
    assert values == [pytest.approx(500.0)] * len(VALID_PSELFIES)  # (400+500+600)/3


def test_ensemble_std_is_zero_for_identical_members_and_positive_otherwise():
    identical = EnsemblePropertyPredictor(
        [_ConstantPredictor(500.0), _ConstantPredictor(500.0)]
    )
    mean, std, n_members = identical.predict_with_uncertainty(["[At][C][C][At]"])[0]
    assert mean == pytest.approx(500.0)
    assert std == pytest.approx(0.0)
    assert n_members == 2

    differing = EnsemblePropertyPredictor(
        [_ConstantPredictor(480.0), _ConstantPredictor(520.0)]
    )
    mean, std, n_members = differing.predict_with_uncertainty(["[At][C][C][At]"])[0]
    assert mean == pytest.approx(500.0)
    assert std == pytest.approx(20.0)  # population std of {480, 520}
    assert n_members == 2


def test_ensemble_drops_non_finite_members_from_the_mean():
    """A member with no answer leaves the average; it never drags it to zero."""
    ensemble = EnsemblePropertyPredictor(
        [
            _ConstantPredictor(480.0),
            _ConstantPredictor(520.0),
            _ConstantPredictor(float("nan")),
        ]
    )
    mean, std, n_members = ensemble.predict_with_uncertainty(["[At][C][C][At]"])[0]
    assert mean == pytest.approx(500.0)  # NOT (480 + 520 + 0) / 3 = 333.3
    assert std == pytest.approx(20.0)
    assert n_members == 2


def test_ensemble_is_non_finite_only_when_no_member_answered():
    ensemble = EnsemblePropertyPredictor(
        [_ConstantPredictor(float("nan")), _ConstantPredictor(float("nan"))]
    )
    mean, std, n_members = ensemble.predict_with_uncertainty(["[At][C][C][At]"])[0]
    assert math.isnan(mean)
    assert math.isnan(std)
    assert n_members == 0
    assert math.isnan(ensemble(["[At][C][C][At]"])[0])


def test_ensemble_satisfies_the_injected_callable_contract():
    ensemble = EnsemblePropertyPredictor(
        [_ConstantPredictor(490.0), _ConstantPredictor(510.0)]
    )
    report = evaluate_generation(
        VALID_PSELFIES,
        target_property=500.0,
        tolerance=50.0,
        property_predictor=ensemble,
        compute_sa=False,
    )
    assert report.n_property_values == len(report.screened_psmiles)
    assert report.property_mean == pytest.approx(500.0)
    assert report.target_property_rate == pytest.approx(1.0)


def test_ensemble_of_real_predictors(trained_checkpoint):
    """Two members loaded from real checkpoints agree, so their spread is 0."""
    ensemble = EnsemblePropertyPredictor.from_checkpoints(
        [trained_checkpoint, trained_checkpoint], device="cpu"
    )
    summaries = ensemble.predict_with_uncertainty(VALID_PSELFIES)
    assert len(summaries) == len(VALID_PSELFIES)
    for mean, std, n_members in summaries:
        assert mean == pytest.approx(500.0)
        assert std == pytest.approx(0.0)
        assert n_members == 2


def test_ensemble_requires_at_least_one_member():
    with pytest.raises(ValueError):
        EnsemblePropertyPredictor([])
    with pytest.raises(ValueError):
        EnsemblePropertyPredictor.from_checkpoints([])


# ------------------------------------------------------- scripts/run_splits.py
def _load_script(name: str, path: Path):
    """Import a file from scripts/ without putting scripts/ on sys.path."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: @dataclass resolves string annotations
    # through sys.modules[cls.__module__].
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_run_splits():
    """Import ``scripts/run_splits.py``."""
    return _load_script("polyt5_run_splits", RUN_SPLITS_PATH)


def _load_evaluate_generations():
    """Import ``scripts/evaluate_generations.py``."""
    return _load_script("polyt5_evaluate_generations", EVALUATE_GENERATIONS_PATH)


def test_run_splits_builds_five_covering_80_20_splits():
    """Split construction only -- no training happens in this test.

    The paper's protocol is FIVE INDEPENDENT random 80/20 splits (see
    :func:`polyt5.data.splits.make_kfold_random_splits`), not a partitioning
    k-fold, so the five test sets legitimately overlap each other. What must
    hold is per split: train, val and test are pairwise disjoint and together
    cover every index exactly once, and test is 20% of the data.
    """
    run_splits = _load_run_splits()
    n = 100
    splits = run_splits.build_splits(n, n_splits=5, train_fraction=0.8,
                                     val_fraction=0.1, base_seed=0)
    assert len(splits) == 5

    for split in splits:
        train, val, test = set(split.train), set(split.val), set(split.test)
        assert train.isdisjoint(test)
        assert train.isdisjoint(val)
        assert val.isdisjoint(test)
        assert train | val | test == set(range(n))
        assert len(split.train) + len(split.val) + len(split.test) == n
        assert len(split.test) == 20  # 20% of 100
        assert len(split.val) == 8  # 10% of the 80-index train pool

    # Independent seeds, so the five splits must not be copies of one another.
    test_sets = [tuple(sorted(split.test)) for split in splits]
    assert len(set(test_sets)) == 5


def test_run_splits_zero_val_fraction_keeps_full_train_pool():
    run_splits = _load_run_splits()
    splits = run_splits.build_splits(50, n_splits=2, train_fraction=0.8,
                                     val_fraction=0.0, base_seed=0)
    assert len(splits) == 2
    for split in splits:
        assert split.val == []
        assert len(split.train) == 40
        assert len(split.test) == 10


# -------------------------------------------------------- conditioning report
def test_conditioning_report_hand_computed():
    """Every statistic checked against arithmetic done by hand.

    errors = predicted - target = [+10, -20, +100, -5]; |errors| = [10, 20, 100, 5].
      mae     = (10 + 20 + 100 + 5) / 4            = 33.75
      median  = median(5, 10, 20, 100)             = (10 + 20) / 2 = 15.0
      rmse    = sqrt((100 + 400 + 10000 + 25) / 4) = sqrt(2631.25)
      bias    = (10 - 20 + 100 - 5) / 4            = 21.25
      within 25 / 50 / 100 K                       = 3/4, 3/4, 4/4
    """
    report = conditioning_report(
        [510.0, 480.0, 600.0, 495.0],
        [500.0, 500.0, 500.0, 500.0],
        tolerances=[25.0, 50.0, 100.0],
    )
    assert report.n_candidates == 4
    assert report.n_scored == 4
    assert report.n_missing_target == 0
    assert report.n_non_numeric_prediction == 0
    assert report.mae == pytest.approx(33.75)
    assert report.median_abs_error == pytest.approx(15.0)
    assert report.rmse == pytest.approx(math.sqrt(2631.25))
    assert report.mean_signed_bias == pytest.approx(21.25)
    assert report.within_tolerance == {
        "25": pytest.approx(0.75),
        "50": pytest.approx(0.75),
        "100": pytest.approx(1.0),
    }
    assert report.target_min == 500.0
    assert report.target_max == 500.0
    assert report.target_mean == pytest.approx(500.0)
    assert report.to_dict()["mae"] == pytest.approx(33.75)


def test_signed_bias_separates_drift_from_scatter():
    """The distinction the metric exists for: same MAE, different failure."""
    scattered = conditioning_report([540.0, 460.0], [500.0, 500.0])
    drifting = conditioning_report([460.0, 460.0], [500.0, 500.0])

    assert scattered.mae == pytest.approx(40.0)
    assert drifting.mae == pytest.approx(40.0)
    assert scattered.mean_signed_bias == pytest.approx(0.0)  # random +/- 40 K
    assert drifting.mean_signed_bias == pytest.approx(-40.0)  # uniformly 40 K low


def test_missing_target_is_skipped_and_counted_not_treated_as_zero():
    """A row with no target must not be scored against 0 K."""
    report = conditioning_report([510.0, 480.0, 600.0], [500.0, None, 500.0])
    assert report.n_candidates == 3
    assert report.n_scored == 2
    assert report.n_missing_target == 1
    # errors over the scored pairs are [+10, +100]
    assert report.mae == pytest.approx(55.0)
    assert report.mean_signed_bias == pytest.approx(55.0)
    # Treating the missing target as 0.0 would have made |480 - 0| = 480 an
    # error, dragging MAE to (10 + 480 + 100) / 3 = 196.67.
    assert report.mae != pytest.approx(196.6667)


def test_non_finite_prediction_is_skipped_and_counted():
    report = conditioning_report(
        [510.0, float("nan"), 600.0], [500.0, 500.0, 500.0]
    )
    assert report.n_scored == 2
    assert report.n_non_numeric_prediction == 1
    assert report.n_missing_target == 0
    assert report.mae == pytest.approx(55.0)


def test_conditioning_report_is_all_none_when_nothing_is_scorable():
    report = conditioning_report([float("nan")], [None])
    assert report.n_scored == 0
    assert report.mae is None
    assert report.median_abs_error is None
    assert report.rmse is None
    assert report.mean_signed_bias is None  # never 0.0, which would look perfect
    assert report.within_tolerance == {}
    assert report.target_min is None


def test_conditioning_report_tolerance_ladder_is_cleaned():
    report = conditioning_report(
        [505.0], [500.0], tolerances=[100.0, 25.0, 25.0, -5.0, 0.0, float("nan")]
    )
    assert report.tolerances == [25.0, 100.0]  # deduped, sorted, non-positive dropped
    assert set(report.within_tolerance) == {"25", "100"}


def test_conditioning_report_rejects_misaligned_sequences():
    """A length mismatch scores each candidate against somebody else's target."""
    with pytest.raises(ValueError, match="same length"):
        conditioning_report([1.0, 2.0], [1.0])


# ------------------------------------------- scripts/evaluate_generations.py
def _write_generations(path: Path, rows: Sequence[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def test_read_generations_parses_targets_and_records_missing(tmp_path):
    module = _load_evaluate_generations()
    path = _write_generations(tmp_path / "generations.jsonl", [
        {"target_property": "500.0", "generated": VALID_PSELFIES[0]},
        {"target_property": 412.5, "generated": VALID_PSELFIES[1]},
        {"generated": VALID_PSELFIES[2]},
        {"target_property": "not a number", "generated": VALID_PSELFIES[3]},
    ])
    generated, targets = module.read_generations(path)
    assert generated == VALID_PSELFIES
    # Missing and unparseable targets are None -- never 0.0.
    assert targets == [500.0, 412.5, None, None]


def test_choose_mode_prefers_per_row_and_target_forces_fixed():
    module = _load_evaluate_generations()
    assert module.choose_mode([500.0, 400.0], None) == module.MODE_PER_ROW
    # --target is an explicit override even when rows carry their own targets.
    assert module.choose_mode([500.0, 400.0], 500.0) == module.MODE_FIXED
    # Nothing to be per-row about.
    assert module.choose_mode([None, None], None) == module.MODE_FIXED


def test_tolerances_flag_is_parsed():
    module = _load_evaluate_generations()
    args = module.parse_args([
        "--generations", "g.jsonl", "--out", "o", "--tolerances", "10", "20", "30",
    ])
    assert args.tolerances == [10.0, 20.0, 30.0]
    assert args.target is None  # per-row is the default
    defaults = module.parse_args(["--generations", "g.jsonl", "--out", "o"])
    assert defaults.tolerances == [25.0, 50.0, 100.0]


def test_scored_candidates_maps_survivors_back_to_their_rows():
    module = _load_evaluate_generations()
    candidates = [VALID_PSELFIES[0], "utter garbage", VALID_PSELFIES[1]]
    report = evaluate_generation(candidates, compute_sa=False)
    rows, strings = module.scored_candidates(report, "pv", "canonical")
    assert rows == [0, 2]  # row 1 failed the screen and is not scored
    # Stage "pv" with canonical input must reproduce the report's screened set.
    assert strings == report.screened_psmiles


def test_scored_candidates_stage_widens_the_population():
    """A looser stage is a larger, easier population -- and must be named."""
    module = _load_evaluate_generations()
    # "[At][C][C][At][C][At]" decodes to a molecule with three [At] termini:
    # RDKit-valid (SV) but a PV failure.
    candidates = [VALID_PSELFIES[0], "[At][C][C][At][C][At]"]
    report = evaluate_generation(candidates, compute_sa=False)

    pv_rows, _ = module.scored_candidates(report, "pv")
    sv_rows, _ = module.scored_candidates(report, "sv")
    assert pv_rows == [0]
    assert sv_rows == [0, 1]
    assert set(pv_rows).issubset(sv_rows)

    with pytest.raises(ValueError, match="stage must be one of"):
        module.scored_candidates(report, "nonsense")


def test_score_input_raw_preserves_what_the_model_emitted():
    """The canonical round trip rewrites the SELFIES; raw does not.

    ``[At][O][C][C][At]`` canonicalises to ``[At]CCO[At]``, which re-encodes to
    ``[At][C][C][O][At]`` -- a different string from the one generated. An RLVR
    reward must score the string the policy actually produced.
    """
    module = _load_evaluate_generations()
    generated = "[At][O][C][C][At]"
    report = evaluate_generation([generated], compute_sa=False)

    _, raw = module.scored_candidates(report, "pv", "raw")
    _, canonical = module.scored_candidates(report, "pv", "canonical")
    assert raw == [generated]
    assert canonical == ["[At]CCO[At]"]
    assert raw != canonical

    with pytest.raises(ValueError, match="source must be"):
        module.scored_candidates(report, "pv", "nonsense")


def test_score_stage_flag_is_recorded(tmp_path, trained_checkpoint):
    """The denominator choice ends up in evaluation.json, never implicit."""
    module = _load_evaluate_generations()
    generations = _write_generations(tmp_path / "generations.jsonl", [
        {"target_property": "500.0", "generated": pselfies} for pselfies in VALID_PSELFIES
    ])
    out_dir = tmp_path / "out"
    assert module.main([
        "--generations", str(generations),
        "--predictor-checkpoint", str(trained_checkpoint),
        "--out", str(out_dir), "--device", "cpu", "--no-sa",
        "--score-stage", "sv",
    ]) == 0
    payload = json.loads((out_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert payload["score_stage"] == "sv"
    assert payload["score_input"] == "raw"  # the default: what the model emitted
    assert payload["n_score_stage_candidates"] == 4
    assert payload["conditioning"]["n_scored"] == 4
    assert "CANONICAL PSMILES" in payload["note_property_stats_input"]


def test_circularity_is_reported_even_when_provenance_is_unknown(tmp_path):
    module = _load_evaluate_generations()
    generations = _write_generations(tmp_path / "generations.jsonl", [
        {"target_property": "500.0", "generated": VALID_PSELFIES[0]},
    ])
    assessment = module.assess_circularity(generations, [tmp_path / "missing.pt"])
    assert assessment["status"] == "unknown_provenance"
    assert "CIRCULARITY" in assessment["warning"]

    # No predictor at all: nothing to be circular about.
    none_used = module.assess_circularity(generations, [])
    assert none_used["status"] == "not_applicable"
    assert none_used["warning"] is None


def test_circularity_detects_a_shared_dataset(tmp_path, tokenizer):
    """Same data.csv_path on both sides is the self-referential case."""
    module = _load_evaluate_generations()
    run_dir = tmp_path / "generation_run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(
        "data:\n  csv_path: data/external/LAMALAB_CURATED_Tg.csv\n", encoding="utf-8"
    )
    generations = _write_generations(run_dir / "generations.jsonl", [
        {"target_property": "500.0", "generated": VALID_PSELFIES[0]},
    ])

    seed_everything(0)
    model = PolyT5ForConditionalGeneration(_tiny_config(tokenizer))
    checkpoint = tmp_path / "predictor.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        epoch=0,
        global_step=0,
        config={"data": {"csv_path": "data/external/LAMALAB_CURATED_Tg.csv"}},
        model_config=model.config.to_dict(),
        tokenizer_sha256=tokenizer.sha256,
    )
    assessment = module.assess_circularity(generations, [checkpoint])
    assert assessment["status"] == "shared_dataset"
    assert "self-referential" in assessment["warning"]

    # A predictor trained on a different corpus is not the same-corpus case.
    other = tmp_path / "other.pt"
    save_checkpoint(
        other,
        model=model,
        epoch=0,
        global_step=0,
        config={"data": {"csv_path": "data/external/SOMETHING_ELSE.csv"}},
        model_config=model.config.to_dict(),
        tokenizer_sha256=tokenizer.sha256,
    )
    assert module.assess_circularity(generations, [other])["status"] == "distinct_datasets"


def test_per_row_mode_end_to_end(tmp_path, trained_checkpoint):
    """The whole CLI in per-row mode, against a file with known targets.

    The predictor emits 500.0 for everything and every row asks for 500.0, so
    the conditioning error must be exactly zero and every tolerance a full hit.
    The paper's fixed-window TP must be reported as NOT MEASURED.
    """
    module = _load_evaluate_generations()
    generations = _write_generations(tmp_path / "generations.jsonl", [
        {"target_property": "500.0", "generated": pselfies} for pselfies in VALID_PSELFIES
    ])
    out_dir = tmp_path / "out"
    assert module.main([
        "--generations", str(generations),
        "--predictor-checkpoint", str(trained_checkpoint),
        "--out", str(out_dir),
        "--device", "cpu",
        "--no-sa",
        "--tolerances", "10", "50",
    ]) == 0

    payload = json.loads((out_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "per_row"
    assert payload["n_rows_with_target"] == 4
    assert payload["row_target_min"] == 500.0
    assert payload["row_target_max"] == 500.0
    # No fixed window was applied, so TP is honestly absent.
    assert payload["report"]["target_property_rate"] is None
    assert "per-row mode" in payload["tp_not_measured_because"]

    conditioning = payload["conditioning"]
    assert conditioning["n_scored"] == payload["report"]["counts"]["n_pv"] == 4
    assert conditioning["n_missing_target"] == 0
    assert conditioning["mae"] == pytest.approx(0.0)
    assert conditioning["median_abs_error"] == pytest.approx(0.0)
    assert conditioning["mean_signed_bias"] == pytest.approx(0.0)
    assert conditioning["within_tolerance"] == {"10": 1.0, "50": 1.0}
    assert conditioning["tolerances"] == [10.0, 50.0]
    assert payload["circularity"]["warning"]  # always recorded


def test_per_row_mode_records_rows_without_a_target(tmp_path, trained_checkpoint):
    """A row missing target_property is skipped and counted, not scored as 0 K."""
    module = _load_evaluate_generations()
    rows = [{"target_property": "500.0", "generated": p} for p in VALID_PSELFIES[:3]]
    rows.append({"generated": VALID_PSELFIES[3]})  # no target_property
    generations = _write_generations(tmp_path / "generations.jsonl", rows)
    out_dir = tmp_path / "out"
    assert module.main([
        "--generations", str(generations),
        "--predictor-checkpoint", str(trained_checkpoint),
        "--out", str(out_dir), "--device", "cpu", "--no-sa",
    ]) == 0

    payload = json.loads((out_dir / "evaluation.json").read_text(encoding="utf-8"))
    conditioning = payload["conditioning"]
    assert payload["n_rows_with_target"] == 3
    assert conditioning["n_candidates"] == 4
    assert conditioning["n_scored"] == 3
    assert conditioning["n_missing_target"] == 1
    assert conditioning["mae"] == pytest.approx(0.0)  # unaffected by the skipped row


def test_fixed_mode_end_to_end(tmp_path, trained_checkpoint):
    """--target forces the paper's fixed-window measurement."""
    module = _load_evaluate_generations()
    generations = _write_generations(tmp_path / "generations.jsonl", [
        {"target_property": "300.0", "generated": pselfies} for pselfies in VALID_PSELFIES
    ])
    out_dir = tmp_path / "out"
    assert module.main([
        "--generations", str(generations),
        "--predictor-checkpoint", str(trained_checkpoint),
        "--out", str(out_dir), "--device", "cpu", "--no-sa",
        "--target", "500", "--tolerance", "50",
    ]) == 0

    payload = json.loads((out_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "fixed"
    assert payload["target_property"] == 500.0
    # Every prediction is 500.0, so all of them sit in the 500 +/- 50 K window.
    assert payload["report"]["target_property_rate"] == pytest.approx(1.0)
    # The rows actually asked for 300 K, and the conditioning report says so:
    # 500 - 300 = +200 K on every candidate.
    assert payload["conditioning"]["mae"] == pytest.approx(200.0)
    assert payload["conditioning"]["mean_signed_bias"] == pytest.approx(200.0)


# ------------------------------------------------------------ layering guards
def test_evaluation_layer_stays_free_of_torch_and_of_the_predictor():
    """polyt5.evaluation must not drag torch (or this package) into a process."""
    script = (
        "import sys; import polyt5.evaluation; "
        "assert 'torch' not in sys.modules, "
        "sorted(m for m in sys.modules if m.startswith('torch')); "
        "assert 'polyt5.inference' not in sys.modules, 'evaluation imported the predictor'"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert completed.returncode == 0, completed.stderr
