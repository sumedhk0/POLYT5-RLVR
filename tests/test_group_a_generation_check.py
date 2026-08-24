# tests/test_group_a_generation_check.py
"""Group A Task 14: a prediction gain bought with a generation loss is a trade."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polyt5.evaluation.filters import FilterCounts
from polyt5.evaluation.generation_metrics import GenerationReport
from polyt5.evaluation.generation_regression import (
    DEFAULT_REGRESSION_TOLERANCE,
    check_generation_regression,
    load_arm_b_generation_baseline,
    requires_generation_check,
)

BASELINE_PV = 0.558
BASELINE_TP = 0.5878


def make_report(pv_rate: float, tp_rate: float | None) -> GenerationReport:
    n_input = 1000
    return GenerationReport(
        counts=FilterCounts(n_input=n_input, n_sv=n_input, n_tsd=n_input, n_dd=n_input,
                            n_pv=round(pv_rate * n_input)),
        sr_rate=1.0, n_unique=n_input, n_novel=n_input, duplicate_rate=0.0,
        sa_available=True, n_sa_scored=n_input, sa_mean=3.0, sa_median=3.0,
        sa_fraction_above_6=0.0,
        property_target=500.0, property_tolerance=50.0, n_property_values=n_input,
        property_mean=500.0, property_median=500.0, property_std=10.0,
        target_property_rate=tp_rate,
        fingerprints_available=True, diversity={},
    )


def test_only_the_arms_that_share_the_encoder_with_generation_need_the_check():
    assert requires_generation_check({"multitask": True}) is True
    assert requires_generation_check({"multitask": False, "regression_head": True}) is False
    assert requires_generation_check({}) is False


def test_matching_the_baseline_is_not_a_regression():
    check = check_generation_regression(
        make_report(BASELINE_PV, BASELINE_TP), arm="A5",
        baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP,
    )
    assert check.regressed is False
    assert check.pv_delta == pytest.approx(0.0, abs=1e-6)
    assert check.verdict == "no regression"


def test_a_drop_beyond_the_tolerance_is_a_regression():
    check = check_generation_regression(
        make_report(0.40, BASELINE_TP), arm="A6",
        baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP,
    )
    assert check.regressed is True
    assert check.pv_delta < 0
    assert "regress" in check.verdict


def test_a_drop_inside_the_tolerance_is_not_a_regression():
    check = check_generation_regression(
        make_report(BASELINE_PV - 0.01, BASELINE_TP), arm="A5",
        baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP,
    )
    assert check.regressed is False
    assert check.tolerance == DEFAULT_REGRESSION_TOLERANCE


def test_a_tp_drop_alone_is_enough_to_fire():
    check = check_generation_regression(
        make_report(BASELINE_PV, 0.40), arm="A6",
        baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP,
    )
    assert check.regressed is True


def test_an_improvement_is_reported_as_such_not_hidden():
    check = check_generation_regression(
        make_report(0.70, 0.70), arm="A5",
        baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP,
    )
    assert check.regressed is False
    assert check.pv_delta > 0
    assert check.verdict == "improved"


def test_a_missing_tp_rate_is_none_and_does_not_read_as_zero():
    check = check_generation_regression(
        make_report(BASELINE_PV, None), arm="A5",
        baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP,
    )
    assert check.tp_rate is None
    assert check.tp_delta is None
    assert check.regressed is False, "an unmeasured TP is not a failed TP"


def test_a_nonpositive_tolerance_is_refused():
    with pytest.raises(ValueError, match="tolerance"):
        check_generation_regression(
            make_report(BASELINE_PV, BASELINE_TP), arm="A5",
            baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP, tolerance=0.0,
        )


def test_the_baseline_rates_come_from_the_frozen_artifact(tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text(
        json.dumps({"arm_b_tuned_sampling": {"pv_rate": 0.558, "tp_rate": 0.5878}}),
        encoding="utf-8",
    )
    assert load_arm_b_generation_baseline(path) == (0.558, 0.5878)


def test_an_artifact_without_arm_b_is_refused(tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps({"arm_a_default_sampling": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="arm_b_tuned_sampling"):
        load_arm_b_generation_baseline(path)


def test_the_real_frozen_artifact_still_says_pv_558_and_tp_588():
    repo = Path(__file__).resolve().parents[1]
    artifact = repo / "artifacts" / "baseline" / "frozen_baseline.json"
    if not artifact.is_file():
        pytest.skip("frozen baseline artifact missing")
    pv_rate, tp_rate = load_arm_b_generation_baseline(artifact)
    assert pv_rate == pytest.approx(0.558, abs=1e-4)
    assert tp_rate == pytest.approx(0.5878, abs=1e-4)


def test_the_check_serialises_for_the_run_directory():
    payload = check_generation_regression(
        make_report(0.50, 0.50), arm="A6",
        baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP,
    ).to_dict()
    assert payload["arm"] == "A6"
    assert payload["regressed"] is True
    assert payload["baseline_pv_rate"] == pytest.approx(BASELINE_PV)


def test_just_inside_tolerance_is_not_a_regression():
    """0.019 below baseline PV, with tolerance 0.02, must not fire."""
    check = check_generation_regression(
        make_report(BASELINE_PV - 0.019, BASELINE_TP), arm="A5",
        baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP,
    )
    assert check.regressed is False


def test_exactly_at_tolerance_is_not_a_regression():
    """A drop of exactly the tolerance is inside the band (inclusive), not beyond it.

    The tolerance is derived from the actual reconstructed pv_rate rather than a
    second independent decimal literal: subtracting two independently-rounded
    decimals (e.g. 0.558 - 0.02) does not reliably reproduce -0.02 bit-for-bit in
    IEEE-754 double precision, so a literal-vs-literal boundary test would be
    testing float noise, not the ``<`` vs ``<=`` distinction this exists to catch.
    """
    report = make_report(0.40, BASELINE_TP)
    tolerance = BASELINE_PV - report.counts.pv_rate
    assert -tolerance == report.counts.pv_rate - BASELINE_PV, "must land exactly on the edge"
    check = check_generation_regression(
        report, arm="A5",
        baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP, tolerance=tolerance,
    )
    assert check.regressed is False


def test_exactly_at_tp_tolerance_is_not_a_regression():
    """Same edge-inclusive boundary, exercised through the TP arm of the check."""
    report = make_report(BASELINE_PV, 0.40)
    tolerance = BASELINE_TP - report.target_property_rate
    assert -tolerance == report.target_property_rate - BASELINE_TP, "must land on the edge"
    check = check_generation_regression(
        report, arm="A6",
        baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP, tolerance=tolerance,
    )
    assert check.regressed is False


def test_just_outside_tolerance_is_a_regression():
    """0.021 below baseline PV, with tolerance 0.02, must fire."""
    check = check_generation_regression(
        make_report(BASELINE_PV - 0.021, BASELINE_TP), arm="A5",
        baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP,
    )
    assert check.regressed is True


def test_tp_just_inside_tolerance_is_not_a_regression():
    check = check_generation_regression(
        make_report(BASELINE_PV, BASELINE_TP - 0.019), arm="A6",
        baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP,
    )
    assert check.regressed is False


def test_tp_just_outside_tolerance_is_a_regression():
    check = check_generation_regression(
        make_report(BASELINE_PV, BASELINE_TP - 0.021), arm="A6",
        baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP,
    )
    assert check.regressed is True


def test_verdict_at_the_boundary_is_not_labelled_improved():
    """Exactly at tolerance is 'no regression', not 'improved' -- it did not gain."""
    report = make_report(0.40, BASELINE_TP)
    tolerance = BASELINE_PV - report.counts.pv_rate
    check = check_generation_regression(
        report, arm="A5",
        baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP, tolerance=tolerance,
    )
    assert check.verdict == "no regression"
