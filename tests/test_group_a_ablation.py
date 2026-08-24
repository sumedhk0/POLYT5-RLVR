"""Group A Task 12: the pre-registered verdict and the matrix that reports it."""

from __future__ import annotations

import pytest

from polyt5.evaluation.ablation import (
    BASELINE_ARM,
    CLAIM_CATEGORY,
    EFFECT_HELPS,
    EFFECT_HURTS,
    EFFECT_NO_EFFECT,
    EFFECT_NOT_RUN,
    ArmResult,
    build_ablation_matrix,
    classify_effect,
    format_ablation_matrix,
    success_threshold,
)
from polyt5.evaluation.regression_metrics import RegressionReport

FROZEN_MEAN = 28.6733
FROZEN_STD = 0.7591


def report(mae: float, rmse: float = 44.0, r2: float = 0.84) -> RegressionReport:
    return RegressionReport(
        n_total=1471, n_valid_numeric=1471, n_non_numeric=0, non_numeric_rate=0.0,
        mae=mae, rmse=rmse, r2=r2, pearson_r=0.92,
    )


def arm(name: str, maes, switches=None) -> ArmResult:
    return ArmResult.from_reports(
        name, switches or {"regression_head": name == "A1"}, [report(m) for m in maes]
    )


def test_the_threshold_is_the_baseline_minus_one_standard_deviation():
    assert success_threshold(FROZEN_MEAN, FROZEN_STD) == pytest.approx(27.9142, abs=1e-4)


def test_a_gain_inside_the_baseline_spread_is_no_effect_not_a_small_win():
    assert classify_effect(28.2, baseline_mean=FROZEN_MEAN, baseline_std=FROZEN_STD) == (
        EFFECT_NO_EFFECT
    )
    assert classify_effect(28.6733, baseline_mean=FROZEN_MEAN, baseline_std=FROZEN_STD) == (
        EFFECT_NO_EFFECT
    )


def test_a_mae_below_the_threshold_helps():
    assert classify_effect(27.0, baseline_mean=FROZEN_MEAN, baseline_std=FROZEN_STD) == (
        EFFECT_HELPS
    )


def test_a_mae_above_the_upper_spread_hurts():
    assert classify_effect(31.0, baseline_mean=FROZEN_MEAN, baseline_std=FROZEN_STD) == (
        EFFECT_HURTS
    )


def test_an_unrun_configuration_is_not_run_not_zero():
    assert classify_effect(None, baseline_mean=FROZEN_MEAN, baseline_std=FROZEN_STD) == (
        EFFECT_NOT_RUN
    )


def test_arm_result_aggregates_the_five_splits():
    result = arm("A1", [27.0, 27.4, 26.8, 27.2, 27.1])
    assert result.n_splits == 5
    assert result.mae_mean == pytest.approx(27.1, abs=1e-6)
    assert result.mae_std is not None
    assert result.non_numeric_rate_mean == pytest.approx(0.0)


def test_arm_result_with_no_reports_carries_none_metrics():
    result = ArmResult.from_reports("A5", {"multitask": True}, [])
    assert result.n_splits == 0
    assert result.mae_mean is None
    assert result.rmse_mean is None


def test_the_matrix_reports_every_arm_including_the_unrun_ones():
    matrix = build_ablation_matrix(
        [arm("B0", [28.7]), arm("A1", [27.0]), ArmResult.from_reports("A2", {}, [])],
        baseline_mean=FROZEN_MEAN, baseline_std=FROZEN_STD,
    )
    assert [row["arm"] for row in matrix["rows"]] == ["B0", "A1", "A2"]
    assert matrix["rows"][2]["effect"] == EFFECT_NOT_RUN
    assert matrix["rows"][2]["mae_mean"] is None


def test_the_matrix_stamps_the_claim_category_and_the_comparison_point():
    matrix = build_ablation_matrix([arm("A1", [27.0])], baseline_mean=FROZEN_MEAN,
                                   baseline_std=FROZEN_STD)
    assert matrix["claim_category"] == CLAIM_CATEGORY
    assert matrix["baseline_arm"] == BASELINE_ARM
    assert matrix["baseline_mae_mean"] == pytest.approx(FROZEN_MEAN)
    assert matrix["success_threshold"] == pytest.approx(27.9142, abs=1e-4)
    assert "paper" in matrix["claim_note"].lower()


def test_the_matrix_refuses_a_duplicate_arm():
    with pytest.raises(ValueError, match="duplicate"):
        build_ablation_matrix([arm("A1", [27.0]), arm("A1", [26.0])],
                              baseline_mean=FROZEN_MEAN, baseline_std=FROZEN_STD)


def test_the_matrix_refuses_a_nonpositive_baseline_spread():
    with pytest.raises(ValueError, match="baseline_std"):
        build_ablation_matrix([arm("A1", [27.0])], baseline_mean=FROZEN_MEAN, baseline_std=0.0)


def test_a_hurting_arm_is_rendered_as_prominently_as_a_helping_one():
    """Spec 6: 'reported with the same prominence'. Row order is arm order, not
    a leaderboard, and no arm is elided."""
    matrix = build_ablation_matrix(
        [arm("A1", [27.0]), arm("A2", [33.0]), arm("A3", [28.4])],
        baseline_mean=FROZEN_MEAN, baseline_std=FROZEN_STD,
    )
    rendered = format_ablation_matrix(matrix)
    assert [row["arm"] for row in matrix["rows"]] == ["A1", "A2", "A3"]
    for arm_id, verdict in (("A1", EFFECT_HELPS), ("A2", EFFECT_HURTS),
                            ("A3", EFFECT_NO_EFFECT)):
        line = next(line for line in rendered.splitlines() if line.strip().startswith(arm_id))
        assert verdict in line


def test_the_rendered_table_names_the_threshold_and_the_claim_category():
    matrix = build_ablation_matrix([arm("A1", [27.0])], baseline_mean=FROZEN_MEAN,
                                   baseline_std=FROZEN_STD)
    rendered = format_ablation_matrix(matrix)
    assert "27.91" in rendered
    assert CLAIM_CATEGORY in rendered
    assert "non_numeric" in rendered


def test_evaluation_package_still_imports_without_torch():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import polyt5.evaluation; sys.exit(1 if 'torch' in sys.modules else 0)"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[:500]


# --- Boundary tests -------------------------------------------------------
# The brief's own tests above only exercise classify_effect far from the
# threshold (27.0 vs 27.9142, 31.0 vs 29.4325). A verdict function that used
# `<=`/`>=` instead of `<`/`>`, or compared against the wrong side of the
# spread, would pass every test above while getting the boundary wrong -- and
# the boundary is exactly where real ablation results land. These pin the
# strict inequalities at both edges of "no effect".

THRESHOLD = FROZEN_MEAN - FROZEN_STD  # 27.9142
UPPER_SPREAD = FROZEN_MEAN + FROZEN_STD  # 29.4324


def test_a_mae_exactly_at_the_threshold_is_no_effect_not_helps():
    """`< threshold` is required, not `<=`: landing exactly on 27.9142 must not
    count as helping -- the threshold is a strict improvement bar."""
    assert classify_effect(THRESHOLD, baseline_mean=FROZEN_MEAN, baseline_std=FROZEN_STD) == (
        EFFECT_NO_EFFECT
    )


def test_a_mae_one_thousandth_below_the_threshold_helps():
    assert classify_effect(
        THRESHOLD - 0.001, baseline_mean=FROZEN_MEAN, baseline_std=FROZEN_STD
    ) == EFFECT_HELPS


def test_a_mae_one_thousandth_above_the_threshold_is_no_effect():
    assert classify_effect(
        THRESHOLD + 0.001, baseline_mean=FROZEN_MEAN, baseline_std=FROZEN_STD
    ) == EFFECT_NO_EFFECT


def test_a_mae_exactly_at_the_upper_spread_is_no_effect_not_hurts():
    """`> baseline + std` is required, not `>=`: landing exactly on the upper
    edge of the baseline's own spread must not be reported as harm."""
    assert classify_effect(UPPER_SPREAD, baseline_mean=FROZEN_MEAN, baseline_std=FROZEN_STD) == (
        EFFECT_NO_EFFECT
    )


def test_a_mae_one_thousandth_above_the_upper_spread_hurts():
    assert classify_effect(
        UPPER_SPREAD + 0.001, baseline_mean=FROZEN_MEAN, baseline_std=FROZEN_STD
    ) == EFFECT_HURTS
