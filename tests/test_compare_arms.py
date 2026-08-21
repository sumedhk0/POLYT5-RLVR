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
    _auditor_predictions,
    _metric_column,
    _score_with_arm,
    apply_success_criterion,
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


def test_arm_metric_accuracy_and_validity_unchanged():
    assert ARM_METRIC["accuracy"] == ("property_mae", "lower")
    assert ARM_METRIC["validity"] == ("pv_rate", "higher")


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


# ------------------------------------------------------------------ _metric_column


def test_metric_column_structural_metric_ignores_predictor():
    assert _metric_column("pv_rate", "auditor") == "pv_rate"
    assert _metric_column("pv_rate", "ensemble") == "pv_rate"


def test_metric_column_non_structural_suffixes_by_predictor():
    assert _metric_column("property_mae", "auditor") == "property_mae_auditor"
    assert _metric_column("composite_score", "ensemble") == "composite_score_ensemble"


# ----------------------------------------------------------- apply_success_criterion


def _row(arm, **values):
    row = dict.fromkeys(MATRIX_COLUMNS)
    row["arm"] = arm
    row.update(values)
    return row


def test_apply_success_criterion_true_when_both_clauses_hold_lower_is_better():
    rows = [
        _row("arm_b", property_mae_ensemble=50.0, property_mae_auditor=50.0),
        _row("accuracy", property_mae_ensemble=40.0, property_mae_auditor=40.0),
    ]
    apply_success_criterion(rows)
    accuracy = rows[1]
    assert accuracy["beats_arm_b"] is True
    assert accuracy["survives_audit"] is True
    assert accuracy["success"] is True


def test_apply_success_criterion_false_when_ensemble_clause_fails_lower_is_better():
    rows = [
        _row("arm_b", property_mae_ensemble=50.0, property_mae_auditor=50.0),
        _row("accuracy", property_mae_ensemble=60.0, property_mae_auditor=40.0),
    ]
    apply_success_criterion(rows)
    accuracy = rows[1]
    assert accuracy["beats_arm_b"] is False
    assert accuracy["survives_audit"] is True
    assert accuracy["success"] is False


def test_apply_success_criterion_false_when_only_audit_clause_fails():
    """The reward-hacking scenario the two-clause design exists to catch: a
    win under the ensemble that does not survive independent auditing.
    """
    rows = [
        _row("arm_b", property_mae_ensemble=50.0, property_mae_auditor=50.0),
        _row("accuracy", property_mae_ensemble=40.0, property_mae_auditor=60.0),
    ]
    apply_success_criterion(rows)
    accuracy = rows[1]
    assert accuracy["beats_arm_b"] is True
    assert accuracy["survives_audit"] is False
    assert accuracy["success"] is False


def test_apply_success_criterion_higher_is_better_direction():
    rows = [
        _row("arm_b", constraint_satisfaction_rate_ensemble=0.3,
             constraint_satisfaction_rate_auditor=0.3),
        _row("constraint", constraint_satisfaction_rate_ensemble=0.5,
             constraint_satisfaction_rate_auditor=0.5),
    ]
    apply_success_criterion(rows)
    assert rows[1]["success"] is True


def test_apply_success_criterion_a_tie_does_not_count_as_beating():
    rows = [
        _row("arm_b", property_mae_ensemble=50.0, property_mae_auditor=50.0),
        _row("accuracy", property_mae_ensemble=50.0, property_mae_auditor=50.0),
    ]
    apply_success_criterion(rows)
    accuracy = rows[1]
    assert accuracy["beats_arm_b"] is False
    assert accuracy["survives_audit"] is False
    assert accuracy["success"] is False


def test_apply_success_criterion_none_propagates_when_metric_unmeasured():
    rows = [
        _row("arm_b", property_mae_ensemble=None, property_mae_auditor=50.0),
        _row("accuracy", property_mae_ensemble=40.0, property_mae_auditor=40.0),
    ]
    apply_success_criterion(rows)
    accuracy = rows[1]
    assert accuracy["beats_arm_b"] is None
    assert accuracy["success"] is None


def test_apply_success_criterion_none_for_arms_with_no_optimized_metric():
    rows = [
        _row("arm_a"),
        _row("arm_b", property_mae_ensemble=50.0, property_mae_auditor=50.0),
    ]
    apply_success_criterion(rows)
    assert rows[0]["beats_arm_b"] is None
    assert rows[0]["survives_audit"] is None
    assert rows[0]["success"] is None


def test_apply_success_criterion_structural_metric_reports_na_not_true():
    """Ruling D / finding 7: ``pv_rate`` is structural, so clause 2 must
    never silently read as an independently-confirmed pass.
    """
    rows = [
        _row("arm_b", pv_rate=0.5),
        _row("validity", pv_rate=0.6),
    ]
    apply_success_criterion(rows)
    validity = rows[1]
    assert validity["beats_arm_b"] is True
    assert validity["survives_audit"] == "N/A - structural metric, no predictor involved"
    assert validity["success"] is True  # reduces to beats_arm_b alone


def test_apply_success_criterion_structural_metric_false_when_it_does_not_beat_arm_b():
    rows = [
        _row("arm_b", pv_rate=0.5),
        _row("validity", pv_rate=0.4),
    ]
    apply_success_criterion(rows)
    assert rows[1]["success"] is False


def test_apply_success_criterion_regression_arm_b_optimized_value_is_never_read():
    """The exact bug fixed by hand in the original submission: arm_b has no
    ``ARM_METRIC`` entry, so ``arm_b["optimized_value_*"]`` is ALWAYS
    ``None`` on its own row. Reading it (instead of arm_b's own
    ``property_mae_*`` column) makes every RLVR arm's success verdict
    ``None`` regardless of the real numbers -- this must not regress.
    """
    rows = [
        _row("arm_b", property_mae_ensemble=50.0, property_mae_auditor=50.0,
             optimized_value_ensemble=None, optimized_value_auditor=None),
        _row("accuracy", property_mae_ensemble=30.0, property_mae_auditor=30.0),
    ]
    apply_success_criterion(rows)
    accuracy = rows[1]
    assert accuracy["success"] is True, (
        "success came back non-True even though accuracy clearly beats arm_b on both "
        "predictors -- the arm_b['optimized_value_*'] regression is back"
    )


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

    composite_arm = _RecordingArm()
    constraint_arm = _RecordingArm()

    row = compare_arms.evaluate_arm(
        None, None, arm_key="composite", label="test", kind="rlvr",
        point=SweepPoint(temperature=0.7, top_p=0.95), targets=[300.0, 400.0], n_samples=2,
        training_index=None, auditor=fake_auditor, ensemble=_FakeEnsemble(),
        composite_arm=composite_arm, constraint_arm=constraint_arm, device="cpu",
        batch_size=2, max_length=200, seed=0, tolerance=50.0,
        checkpoint_label=None, checkpoint_sha256=None, novelty_index_sha256=None,
    )

    # evaluate_arm computes the auditor-scored column before the
    # ensemble-scored one, for both arms (see its body) -- calls[0] is
    # therefore always the auditor-built triples, calls[1] the ensemble's.
    assert composite_arm.calls[0] == [(111.0, 0.0, 1), (222.0, 0.0, 1)]
    assert composite_arm.calls[1] == [(111.0, 9.0, 4), (222.0, 8.0, 4)]
    assert constraint_arm.calls[0] == [(111.0, 0.0, 1), (222.0, 0.0, 1)]
    assert constraint_arm.calls[1] == [(111.0, 9.0, 4), (222.0, 8.0, 4)]
    assert row["composite_score_auditor"] == pytest.approx(1.0)


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
