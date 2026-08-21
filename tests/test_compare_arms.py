"""Unit coverage for ``scripts/compare_arms.py``'s pure-function internals.

The comparison matrix is the study's headline output, so every function that
decides what a row MEANS -- not just what it contains -- is tested here on
plain dicts and stub callables. None of this needs a model, a GPU, or a real
checkpoint.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import compare_arms  # noqa: E402
from compare_arms import (  # noqa: E402
    ARM_METRIC,
    MATRIX_COLUMNS,
    STRUCTURAL_METRICS,
    ArmScorers,
    _auditor_predictions,
    _bootstrap_delta,
    _metric_column,
    _score_with_arm,
    apply_success_criterion,
    check_pre_registration,
    hash_reward_configs,
    reward_config_paths,
    skipped_arm_row,
    write_matrix_csv,
    write_matrix_markdown,
)

# -------------------------------------------------------------------- ARM_METRIC


def test_arm_metric_composite_is_composite_score_not_property_mae():
    """Ruling B / mutant (a): reverting this to ``property_mae`` scores a
    THREE-term objective (tg/pv/novelty) by one term, and must be caught.
    """
    assert ARM_METRIC["composite"] == ("composite_score", "higher")


def test_arm_metric_constraint_is_constraint_satisfaction_rate():
    assert ARM_METRIC["constraint"] == ("constraint_satisfaction_rate", "higher")


def test_arm_metric_accuracy_is_accuracy_score_not_property_mae():
    """Ruling G / mutant (b). C1's reward is mean(closeness x confidence) over
    the FULL rollout batch; ``property_mae`` is unweighted, unclipped and
    computed only over PV survivors, so the two can move in opposite
    directions. Reverting this line judges C1 on a metric its own reward can
    move against -- the identical defect the ledger already ruled invalidating
    for ``composite``.
    """
    assert ARM_METRIC["accuracy"] == ("accuracy_score", "higher")


def test_arm_metric_validity_unchanged():
    assert ARM_METRIC["validity"] == ("pv_rate", "higher")


def test_property_mae_is_still_reported_for_every_row():
    """It is not C1's success metric, but it IS the paper's headline
    conditioning metric and arm_b's frozen number is one, so it stays in the
    matrix as an off-diagonal column.
    """
    assert "property_mae_auditor" in MATRIX_COLUMNS
    assert "property_mae_ensemble" in MATRIX_COLUMNS


def test_every_arm_metric_column_exists_in_the_matrix():
    """A typo in ``ARM_METRIC`` would resolve to a column absent from
    ``MATRIX_COLUMNS``, which ``row.get(...)`` reads as "not measured"
    (``None``) rather than raising -- the exact silent-failure shape the
    review calls out.
    """
    for arm, (metric, _direction) in ARM_METRIC.items():
        for predictor in ("auditor", "ensemble"):
            column = _metric_column(metric, predictor)
            assert column in MATRIX_COLUMNS, (arm, metric, predictor, column)


# --------------------------------------------------------- pre-registration binding


def _frozen():
    import json
    path = REPO_ROOT / "artifacts" / "baseline" / "frozen_baseline.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_arm_metric_matches_the_frozen_pre_registration():
    """Ruling H / finding S4. The executable criterion used to live entirely in
    ``ARM_METRIC`` -- ordinary mutable Python that nothing bound to the
    pre-registration -- so a post-run edit produced a ``summary.json`` that
    looked equally authoritative. This fails if the code and the frozen record
    ever disagree, in EITHER direction.
    """
    problems = check_pre_registration(_frozen())
    assert problems == [], problems


def test_pre_registration_check_catches_a_changed_metric(monkeypatch):
    """The binding test must actually be able to fail -- verified against the
    exact mutant it guards (reverting accuracy to ``property_mae``).
    """
    monkeypatch.setitem(compare_arms.ARM_METRIC, "accuracy", ("property_mae", "lower"))
    problems = check_pre_registration(_frozen())
    assert problems, "reverting accuracy to property_mae was not detected"
    assert any("accuracy" in problem for problem in problems), problems


def test_pre_registration_check_catches_a_changed_margin(monkeypatch):
    monkeypatch.setitem(compare_arms.ARM_MIN_MARGIN, "composite", 0.0)
    problems = check_pre_registration(_frozen())
    assert any("min_margin" in problem for problem in problems), problems


def test_frozen_record_pins_a_margin_for_every_arm():
    registered = _frozen()["pre_registered_metrics"]
    for arm in ARM_METRIC:
        assert float(registered[arm]["min_margin"]) > 0.0, arm


# -------------------------------------------------- reward config provenance (item 1)


def test_reward_config_paths_covers_exactly_the_three_scorer_arms():
    """``validity`` is deliberately excluded -- its optimized metric
    (``pv_rate``) is structural RDKit chemistry with no ``reward:`` block in
    the loop at all; see :data:`STRUCTURAL_METRICS`.
    """
    paths = reward_config_paths()
    assert set(paths) == {"accuracy", "composite", "constraint"}
    for name, path in paths.items():
        assert path.is_file(), (name, path)
        assert path == REPO_ROOT / "configs" / "rl" / f"{name}.yaml"


def test_hash_reward_configs_matches_sha256_of_file():
    paths = reward_config_paths()
    hashes = hash_reward_configs(paths)
    for name, path in paths.items():
        assert hashes[name] == compare_arms.sha256_of_file(path)
        assert len(hashes[name]) == 64


def test_hash_reward_configs_changes_when_a_config_is_edited(tmp_path):
    """The whole point: an unhashed config can be edited with no trace. This
    pins that editing one DOES change the recorded hash.
    """
    path = tmp_path / "accuracy.yaml"
    path.write_text("reward:\n  sigma0: 17.0\n", encoding="utf-8")
    before = hash_reward_configs({"accuracy": path})

    path.write_text("reward:\n  sigma0: 99.0\n", encoding="utf-8")
    after = hash_reward_configs({"accuracy": path})

    assert before["accuracy"] != after["accuracy"]


def test_reward_config_sha256_columns_exist_for_every_scorer_arm():
    for name in ("accuracy", "composite", "constraint"):
        assert f"{name}_reward_config_sha256" in MATRIX_COLUMNS


# --------------------------------------------------------- drift columns (item 3)


def test_drift_columns_are_in_the_matrix():
    assert "max_tanimoto_mean" in MATRIX_COLUMNS
    assert "near_copy_fraction" in MATRIX_COLUMNS


def test_drift_columns_are_diagnostic_never_a_success_metric():
    """Adjudication (d): novelty is exact-canonical-match, so these two
    columns exist to let a reader catch a one-atom-edit near-copy -- they must
    never become part of what an arm is judged to have succeeded at.
    """
    metric_names = {metric for metric, _direction in ARM_METRIC.values()}
    assert "max_tanimoto_mean" not in metric_names
    assert "near_copy_fraction" not in metric_names
    assert "max_tanimoto_mean" not in STRUCTURAL_METRICS
    assert "near_copy_fraction" not in STRUCTURAL_METRICS


# ------------------------------------------------------------------ _bootstrap_delta


def test_bootstrap_delta_is_signed_so_positive_always_means_better():
    higher, _lo, _hi = _bootstrap_delta([1.0] * 50, [0.0] * 50, direction="higher", n_boot=200)
    lower, _lo2, _hi2 = _bootstrap_delta([1.0] * 50, [2.0] * 50, direction="lower", n_boot=200)
    assert higher == pytest.approx(1.0)
    assert lower == pytest.approx(1.0), "for a 'lower is better' metric, being 1.0 lower is +1.0"


def test_bootstrap_delta_interval_excludes_zero_for_a_clean_separation():
    _delta, low, high = _bootstrap_delta([1.0] * 200, [0.0] * 200, direction="higher", n_boot=400)
    assert low > 0.0 and high > 0.0


def test_bootstrap_delta_interval_includes_zero_for_noise():
    own = [1.0 if i % 2 else 0.0 for i in range(40)]
    base = [1.0 if i % 3 else 0.0 for i in range(40)]
    _delta, low, high = _bootstrap_delta(own, base, direction="higher", n_boot=800)
    assert low < 0.0 < high, (low, high)


def test_bootstrap_delta_is_none_on_an_empty_sample():
    assert _bootstrap_delta([], [1.0], direction="higher") == (None, None, None)
    assert _bootstrap_delta([1.0], [], direction="higher") == (None, None, None)


# ----------------------------------------------------------- apply_success_criterion


def _row(arm, *, samples=None, **values):
    row = dict.fromkeys(MATRIX_COLUMNS)
    row["arm"] = arm
    # Default every row to arm_b's operating point, so a test that is not
    # ABOUT the sampling clause does not accidentally exercise it.
    row["temperature"] = 0.7
    row["top_p"] = 0.95
    row.update(values)
    if samples is not None:
        row[compare_arms.SAMPLES_KEY] = samples
    return row


def _samples(metric, values, *, auditor=None):
    """Per-candidate samples for one metric; same on both predictors unless told."""
    return {metric: {"ensemble": list(values), "auditor": list(auditor or values)}}


N = 300
BOOT = {"n_boot": 300}


def test_apply_success_criterion_true_when_every_clause_holds():
    rows = [
        _row("arm_b", composite_score_ensemble=0.5, composite_score_auditor=0.5,
             samples=_samples("composite_score", [0.5] * N)),
        _row("composite", composite_score_ensemble=0.9, composite_score_auditor=0.9,
             samples=_samples("composite_score", [0.9] * N)),
    ]
    apply_success_criterion(rows, **BOOT)
    composite = rows[1]
    assert composite["beats_arm_b"] is True
    assert composite["survives_audit"] is True
    assert composite["success"] is True
    assert composite["delta_ensemble"] == pytest.approx(0.4)
    assert composite["delta_ensemble_ci_low"] > 0.0


def test_apply_success_criterion_rejects_a_win_smaller_than_the_margin():
    """Ruling H / mutant (c): reverting the criterion to a bare comparison
    makes this pass. A 0.005 improvement is real, perfectly reproducible, and
    has an interval that excludes zero by a mile -- and is a quarter of the
    pre-registered 0.02 minimum effect size, so it is NOT a win.
    """
    rows = [
        _row("arm_b", composite_score_ensemble=0.500, composite_score_auditor=0.500,
             samples=_samples("composite_score", [0.500] * N)),
        _row("composite", composite_score_ensemble=0.505, composite_score_auditor=0.505,
             samples=_samples("composite_score", [0.505] * N)),
    ]
    apply_success_criterion(rows, **BOOT)
    composite = rows[1]
    assert composite["delta_ensemble"] == pytest.approx(0.005)
    assert composite["delta_ensemble_ci_low"] > 0.0, "the CI alone would call this a win"
    assert composite["beats_arm_b"] is False, "below the pre-registered min_margin"
    assert composite["success"] is False


def test_apply_success_criterion_rejects_a_large_win_whose_interval_spans_zero():
    """The other half of Ruling H / mutant (c): a difference FIVE TIMES the
    minimum effect size, in the right direction, that is still
    indistinguishable from sampling noise. The margin alone would call this a
    win; only the interval catches it.
    """
    own = [1.0] * 22 + [0.0] * 18          # mean 0.55
    base = [1.0] * 18 + [0.0] * 22         # mean 0.45
    rows = [
        _row("arm_b", composite_score_ensemble=0.45, composite_score_auditor=0.45,
             samples=_samples("composite_score", base)),
        _row("composite", composite_score_ensemble=0.55, composite_score_auditor=0.55,
             samples=_samples("composite_score", own)),
    ]
    apply_success_criterion(rows, **BOOT)
    composite = rows[1]
    assert composite["delta_ensemble"] == pytest.approx(0.10)
    assert composite["delta_ensemble"] > 0.02, "well past the pre-registered min_margin"
    assert composite["delta_ensemble_ci_low"] < 0.0 < composite["delta_ensemble_ci_high"]
    assert composite["beats_arm_b"] is False, (
        "a 0.10 difference on n=40 with unit-scale spread is noise; the margin alone "
        "would have called it a win"
    )


def test_apply_success_criterion_false_when_only_the_audit_clause_fails():
    """The reward-hacking scenario the two-clause design exists to catch: a
    win under the reward ensemble that does not survive the auditor.
    """
    rows = [
        _row("arm_b", composite_score_ensemble=0.5, composite_score_auditor=0.5,
             samples=_samples("composite_score", [0.5] * N)),
        _row("composite", composite_score_ensemble=0.9, composite_score_auditor=0.4,
             samples=_samples("composite_score", [0.9] * N, auditor=[0.4] * N)),
    ]
    apply_success_criterion(rows, **BOOT)
    composite = rows[1]
    assert composite["beats_arm_b"] is True
    assert composite["survives_audit"] is False
    assert composite["success"] is False


def test_apply_success_criterion_a_tie_does_not_count_as_beating():
    rows = [
        _row("arm_b", samples=_samples("accuracy_score", [0.5] * N)),
        _row("accuracy", samples=_samples("accuracy_score", [0.5] * N)),
    ]
    apply_success_criterion(rows, **BOOT)
    accuracy = rows[1]
    assert accuracy["delta_ensemble"] == pytest.approx(0.0)
    assert accuracy["beats_arm_b"] is False
    assert accuracy["success"] is False


def test_apply_success_criterion_none_propagates_when_metric_unmeasured():
    rows = [
        _row("arm_b"),
        _row("accuracy", samples=_samples("accuracy_score", [0.9] * N)),
    ]
    apply_success_criterion(rows, **BOOT)
    accuracy = rows[1]
    assert accuracy["beats_arm_b"] is None
    assert accuracy["success"] is None


def test_apply_success_criterion_none_for_arms_with_no_optimized_metric():
    rows = [
        _row("arm_a"),
        _row("arm_b", samples=_samples("accuracy_score", [0.5] * N)),
    ]
    apply_success_criterion(rows, **BOOT)
    assert rows[0]["beats_arm_b"] is None
    assert rows[0]["survives_audit"] is None
    assert rows[0]["success"] is None


def test_apply_success_criterion_structural_metric_reports_na_not_true():
    """Ruling D: ``pv_rate`` is structural, so clause 2 must never silently
    read as an independently-confirmed pass.
    """
    rows = [
        _row("arm_b", pv_rate=0.5, samples=_samples("pv_rate", [0.5] * N)),
        _row("validity", pv_rate=0.9, samples=_samples("pv_rate", [0.9] * N)),
    ]
    apply_success_criterion(rows, **BOOT)
    validity = rows[1]
    assert validity["beats_arm_b"] is True
    assert validity["survives_audit"] == compare_arms.STRUCTURAL_AUDIT_NOTE
    assert validity["success"] is True  # reduces to beats_arm_b alone
    assert validity["delta_auditor"] is None, "no auditor comparison was made"


def test_apply_success_criterion_structural_metric_false_when_it_does_not_beat_arm_b():
    rows = [
        _row("arm_b", pv_rate=0.9, samples=_samples("pv_rate", [0.9] * N)),
        _row("validity", pv_rate=0.5, samples=_samples("pv_rate", [0.5] * N)),
    ]
    apply_success_criterion(rows, **BOOT)
    assert rows[1]["success"] is False


def test_apply_success_criterion_regression_arm_b_optimized_value_is_never_read():
    """The exact bug fixed by hand in the original submission: arm_b has no
    ``ARM_METRIC`` entry, so ``arm_b["optimized_value_*"]`` is ALWAYS ``None``
    on its own row. Reading it makes every RLVR arm's verdict ``None``
    regardless of the real numbers -- this must not regress.
    """
    rows = [
        _row("arm_b", optimized_value_ensemble=None, optimized_value_auditor=None,
             samples=_samples("accuracy_score", [0.2] * N)),
        _row("accuracy", samples=_samples("accuracy_score", [0.8] * N)),
    ]
    apply_success_criterion(rows, **BOOT)
    assert rows[1]["success"] is True, (
        "success came back non-True even though accuracy clearly beats arm_b on both "
        "predictors -- the arm_b['optimized_value_*'] regression is back"
    )


# ------------------------------------------------- sampling point (finding 11)


def test_sampling_mismatch_refuses_the_verdict_rather_than_failing_the_arm():
    """Finding 11: an arm trained with ``--set train.temperature=1.0`` is
    resampled at 1.0 and cannot be compared to arm_b on any column. The row
    must be marked and ``success`` refused -- ``None``, not ``False``: the arm
    has not lost, it has not been measured comparably.
    """
    rows = [
        _row("arm_b", samples=_samples("accuracy_score", [0.2] * N)),
        _row("accuracy", temperature=1.0, samples=_samples("accuracy_score", [0.9] * N)),
    ]
    apply_success_criterion(rows, **BOOT)
    accuracy = rows[1]
    assert accuracy["sampling_matches_arm_b"] is False
    assert accuracy["beats_arm_b"] is True, "the comparison is still recorded"
    assert accuracy["success"] is None, "but no verdict is issued"


def test_sampling_match_is_recorded_true_when_the_points_agree():
    rows = [
        _row("arm_b", samples=_samples("accuracy_score", [0.2] * N)),
        _row("accuracy", samples=_samples("accuracy_score", [0.9] * N)),
    ]
    apply_success_criterion(rows, **BOOT)
    assert rows[1]["sampling_matches_arm_b"] is True
    assert rows[1]["success"] is True


def test_sampling_match_is_none_for_an_untrained_arm_placeholder():
    rows = [
        _row("arm_b", samples=_samples("accuracy_score", [0.2] * N)),
        skipped_arm_row("accuracy"),
    ]
    apply_success_criterion(rows, **BOOT)
    assert rows[1]["sampling_matches_arm_b"] is None
    assert rows[1]["success"] is None


# ----------------------------------------------------------------- skipped_arm_row


def test_skipped_arm_row_has_every_matrix_column():
    row = skipped_arm_row("composite")
    assert set(MATRIX_COLUMNS) <= set(row)
    assert row["arm"] == "composite"
    assert row["kind"] == "rlvr"
    assert row["optimized_metric"] == "composite_score"
    assert row["checkpoint"] is None


def test_skipped_arm_row_tracks_matrix_columns_growing():
    """Finding 15: the placeholder must be DERIVED from MATRIX_COLUMNS, not a
    hand-maintained key list that drifts when a column is added -- every
    column ``write_matrix_csv``/``write_matrix_markdown`` will look up must
    exist, whether or not the row carries extra bookkeeping fields (e.g.
    ``label``) beyond them.
    """
    row = skipped_arm_row("accuracy")
    assert set(MATRIX_COLUMNS) <= set(row.keys())


# ------------------------------------------------------------------- _score_with_arm


class _FakeResult:
    def __init__(self, value):
        self.value = value


class _FakeArm:
    """Stub ``ArmReward``: gates every other candidate to 0.0."""

    def __call__(self, candidates, targets, predictions):
        return [_FakeResult(1.0 if i % 2 == 0 else 0.0) for i in range(len(candidates))]


def test_score_with_arm_includes_gated_candidates_in_the_mean():
    value = _score_with_arm(
        _FakeArm(), ["a", "b", "c", "d"], [1.0, 2.0, 3.0, 4.0], [(1.0, 0.0, 1)] * 4
    )
    assert value == pytest.approx(0.5)


def test_score_with_arm_empty_batch_is_zero():
    assert _score_with_arm(_FakeArm(), [], [], []) == 0.0


# --------------------------------------------------------------- _auditor_predictions


def test_auditor_predictions_always_carries_zero_std():
    """Ruling C: a single-model auditor has no ensemble disagreement to
    report; ``std`` must always be ``0.0``, never fabricated.

    Takes already-computed means (not an auditor callable) -- see
    ``test_evaluate_arm_auditor_predictions_are_wired_through_the_helper``
    below for why: ``evaluate_arm`` must build ``auditor_means`` with exactly
    ONE call to the auditor (it also needs the raw means for the screened
    slice), so this helper cannot own the call itself without doubling it.
    """
    predictions = _auditor_predictions([123.4, 567.8])
    assert predictions == [(123.4, 0.0, 1), (567.8, 0.0, 1)]


def test_evaluate_arm_auditor_predictions_are_wired_through_the_helper(monkeypatch):
    """Task 8 review finding: ``evaluate_arm`` used to inline its own
    ``[(float(mean), 0.0, 1) for mean in auditor_means]`` instead of calling
    ``_auditor_predictions``, so a mutant changing that inlined ``0.0`` to a
    nonzero std broke nothing -- ``_auditor_predictions`` was dead code in
    the production path and the whole 869-test suite still passed.

    This exercises ``evaluate_arm`` itself end to end (``screen_candidates``
    is the only thing stubbed out; ``property_columns`` and the reward arms
    run for real) and pins the EXACT ``(mean, std, n)`` triples the arms are
    called with, so it fails if that wiring ever regresses -- see
    ``.superpowers/sdd/2026-08-20-grpo-rlvr/task-9-report.md`` for the mutant
    re-introduction that proves this.
    """
    from polyt5.evaluation.filters import CandidateRecord, FilterCounts
    from polyt5.evaluation.sweep import ScreenedBatch, SweepPoint

    def _record(canonical_psmiles):
        return CandidateRecord(
            raw_pselfies="raw", psmiles=canonical_psmiles, canonical_psmiles=canonical_psmiles,
            passed_sv=True, passed_tsd=True, passed_dd=True, passed_pv=True,
            reproducible=True, failure_stage=None, sa_score=None,
        )

    fixed_batch = ScreenedBatch(
        point=SweepPoint(temperature=0.7, top_p=0.95),
        candidates=["cand0", "cand1"],
        sample_targets=[300.0, 400.0],
        records=[_record("A"), _record("B")],
        counts=FilterCounts(n_input=2, n_sv=2, n_tsd=2, n_dd=2, n_pv=2),
        sr_rate=1.0,
        duplicate_rate=0.0,
        mean_length=5.0,
    )

    def fake_screen_candidates(model, tokenizer, **kwargs):
        return fixed_batch

    monkeypatch.setattr(compare_arms, "screen_candidates", fake_screen_candidates)

    def fake_auditor(candidates):
        return [111.0, 222.0]

    class _FakeEnsemble:
        def predict_with_uncertainty(self, candidates):
            return [(111.0, 9.0, 4), (222.0, 8.0, 4)]

    class _RecordingArm:
        def __init__(self):
            self.calls: list[list[tuple[float, float, int]]] = []

        def __call__(self, candidates, targets, predictions):
            self.calls.append(list(predictions))
            return [_FakeResult(1.0) for _ in candidates]

    auditor_scorers = ArmScorers(
        accuracy=_RecordingArm(), composite=_RecordingArm(), constraint=_RecordingArm())
    ensemble_scorers = ArmScorers(
        accuracy=_RecordingArm(), composite=_RecordingArm(), constraint=_RecordingArm())

    class _NullSimilarityMonitor:
        def observe(self, candidates, ensemble_predictions):
            return {"max_tanimoto_mean": None, "near_copy_fraction": None,
                    "max_tanimoto_p90": None, "max_tanimoto_n": None,
                    "auditor_gap_mean": None, "auditor_gap_signed_mean": None,
                    "auditor_gap_n": None}

    row = compare_arms.evaluate_arm(
        None, None, arm_key="composite", label="test", kind="rlvr",
        point=SweepPoint(temperature=0.7, top_p=0.95), targets=[300.0, 400.0], n_samples=2,
        training_index=None, auditor=fake_auditor, ensemble=_FakeEnsemble(),
        auditor_scorers=auditor_scorers, ensemble_scorers=ensemble_scorers,
        similarity_monitor=_NullSimilarityMonitor(), device="cpu",
        batch_size=2, max_length=200, seed=0, tolerance=50.0,
        checkpoint_label=None, checkpoint_sha256=None, novelty_index_sha256=None,
        reward_config_sha256={"accuracy": "a" * 64, "composite": "b" * 64,
                              "constraint": "c" * 64},
    )

    # The auditor-side scorers receive the (mean, 0.0, 1) triples and the
    # ensemble-side scorers the real ones -- the two sets are kept separate
    # precisely so their `ensemble_size` can differ (1 vs len(ensemble)).
    for scorer in (auditor_scorers.accuracy, auditor_scorers.composite,
                   auditor_scorers.constraint):
        assert scorer.calls[0] == [(111.0, 0.0, 1), (222.0, 0.0, 1)]
    for scorer in (ensemble_scorers.accuracy, ensemble_scorers.composite,
                   ensemble_scorers.constraint):
        assert scorer.calls[0] == [(111.0, 9.0, 4), (222.0, 8.0, 4)]
    assert row["composite_score_auditor"] == pytest.approx(1.0)
    assert row["accuracy_score_ensemble"] == pytest.approx(1.0)
    assert row["accuracy_reward_config_sha256"] == "a" * 64
    assert row["composite_reward_config_sha256"] == "b" * 64
    assert row["constraint_reward_config_sha256"] == "c" * 64
    assert row["max_tanimoto_mean"] is None
    assert row["near_copy_fraction"] is None


def test_evaluate_arm_computes_drift_columns_for_baseline_and_rlvr_rows(monkeypatch):
    """Adjudication (d): arm_a/arm_b never ran under a drift monitor, so
    today these columns exist only in the training log. ``evaluate_arm`` must
    compute them for EVERY row -- baseline (``kind="baseline"``, e.g. arm_a)
    included, not just RLVR rows -- reusing the exact ``ensemble_predictions``
    already computed, not a second predictor call.
    """
    from polyt5.evaluation.filters import CandidateRecord, FilterCounts
    from polyt5.evaluation.sweep import ScreenedBatch, SweepPoint

    def _record(canonical_psmiles):
        return CandidateRecord(
            raw_pselfies="raw", psmiles=canonical_psmiles, canonical_psmiles=canonical_psmiles,
            passed_sv=True, passed_tsd=True, passed_dd=True, passed_pv=True,
            reproducible=True, failure_stage=None, sa_score=None,
        )

    fixed_batch = ScreenedBatch(
        point=SweepPoint(temperature=0.7, top_p=0.95),
        candidates=["cand0", "cand1"],
        sample_targets=[300.0, 400.0],
        records=[_record("A"), _record("B")],
        counts=FilterCounts(n_input=2, n_sv=2, n_tsd=2, n_dd=2, n_pv=2),
        sr_rate=1.0, duplicate_rate=0.0, mean_length=5.0,
    )

    def fake_screen_candidates(model, tokenizer, **kwargs):
        return fixed_batch

    monkeypatch.setattr(compare_arms, "screen_candidates", fake_screen_candidates)

    def fake_auditor(candidates):
        return [111.0, 222.0]

    class _FakeEnsemble:
        def predict_with_uncertainty(self, candidates):
            return [(111.0, 9.0, 4), (222.0, 8.0, 4)]

    class _NullArm:
        def __call__(self, candidates, targets, predictions):
            return [_FakeResult(0.0) for _ in candidates]

    class _RecordingSimilarityMonitor:
        def __init__(self):
            self.calls: list[tuple[list[str], list[tuple[float, float, int]]]] = []

        def observe(self, candidates, ensemble_predictions):
            self.calls.append((list(candidates), list(ensemble_predictions)))
            return {"max_tanimoto_mean": 0.42, "near_copy_fraction": 0.5,
                    "max_tanimoto_p90": 0.9, "max_tanimoto_n": 2.0,
                    "auditor_gap_mean": None, "auditor_gap_signed_mean": None,
                    "auditor_gap_n": None}

    scorers = ArmScorers(accuracy=_NullArm(), composite=_NullArm(), constraint=_NullArm())
    monitor = _RecordingSimilarityMonitor()
    reward_hashes = {"accuracy": "a" * 64, "composite": "b" * 64, "constraint": "c" * 64}

    rows = {}
    for kind, arm_key in (("baseline", "arm_a"), ("rlvr", "accuracy")):
        rows[arm_key] = compare_arms.evaluate_arm(
            None, None, arm_key=arm_key, label="test", kind=kind,
            point=SweepPoint(temperature=0.7, top_p=0.95), targets=[300.0, 400.0], n_samples=2,
            training_index=None, auditor=fake_auditor, ensemble=_FakeEnsemble(),
            auditor_scorers=scorers, ensemble_scorers=scorers, similarity_monitor=monitor,
            device="cpu", batch_size=2, max_length=200, seed=0, tolerance=50.0,
            checkpoint_label=None, checkpoint_sha256=None, novelty_index_sha256=None,
            reward_config_sha256=reward_hashes,
        )

    for arm_key, row in rows.items():
        assert row["max_tanimoto_mean"] == pytest.approx(0.42), arm_key
        assert row["near_copy_fraction"] == pytest.approx(0.5), arm_key

    assert len(monitor.calls) == 2
    assert monitor.calls[0] == (["cand0", "cand1"], [(111.0, 9.0, 4), (222.0, 8.0, 4)])
    assert monitor.calls[1] == monitor.calls[0]


# ------------------------------------------------------------------ matrix writers


def test_write_matrix_csv_round_trips_every_column(tmp_path):
    rows = [_row("arm_a", checkpoint="x.pt"), _row("accuracy", pv_rate=0.5)]
    path = write_matrix_csv(rows, tmp_path / "matrix.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(MATRIX_COLUMNS)
        written = list(reader)
    assert written[0]["arm"] == "arm_a"
    assert written[1]["pv_rate"] == "0.5"


def test_write_matrix_markdown_renders_none_as_na_and_bools_as_yes_no(tmp_path):
    rows = [_row("accuracy", success=True, beats_arm_b=False, checkpoint=None)]
    path = write_matrix_markdown(rows, tmp_path / "matrix.md")
    text = path.read_text(encoding="utf-8")
    assert "| yes " in text or "| yes |" in text
    assert "| no " in text or "| no |" in text
    assert "n/a" in text


def test_write_matrix_markdown_column_order_matches_matrix_columns(tmp_path):
    rows = [_row("accuracy")]
    path = write_matrix_markdown(rows, tmp_path / "matrix.md")
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == "| " + " | ".join(MATRIX_COLUMNS) + " |"


# ------------------------------------------------- main(): missing novelty index


def test_main_aborts_before_loading_any_checkpoint_when_novelty_index_missing(
    tmp_path, monkeypatch, capsys
):
    """Ruling E / mutant (c): a missing novelty index must ABORT (exit 1),
    not warn and continue with TSD silently turned into a no-op.

    Also proves the abort happens BEFORE any checkpoint is touched:
    ``load_verified_model``, ``build_reward_ensemble`` and
    ``PolyT5PropertyPredictor`` are replaced with stubs that raise if called
    at all, so this test would fail loudly (not just slowly) if the ordering
    ever regressed to checking the novelty index after loading checkpoints.
    This still needs the real (checked-in) frozen baseline and tokenizer
    artifacts to reach that far -- no model or ``.pt`` checkpoint is loaded.
    """

    def _must_not_be_called(*_args, **_kwargs):
        raise AssertionError(
            "checkpoint/predictor loading was reached despite the missing novelty index"
        )

    class _PredictorMustNotBeCalled:
        @classmethod
        def from_checkpoint(cls, *args, **kwargs):
            _must_not_be_called()

    monkeypatch.setattr(compare_arms, "load_verified_model", _must_not_be_called)
    monkeypatch.setattr(compare_arms, "build_reward_ensemble", _must_not_be_called)
    monkeypatch.setattr(compare_arms, "PolyT5PropertyPredictor", _PredictorMustNotBeCalled)

    exit_code = compare_arms.main([
        "--arms", "accuracy",
        "--novelty-index", str(tmp_path / "does_not_exist"),
        "--out", str(tmp_path / "out"),
        "--device", "cpu",
    ])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "novelty index" in captured.err.lower()
    assert "--allow-missing-novelty-index" in captured.err
    assert not (tmp_path / "out" / "matrix.csv").exists()
