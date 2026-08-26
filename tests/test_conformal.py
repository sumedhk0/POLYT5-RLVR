"""Tests for split-conformal intervals (Phase 4 Group B)."""

from __future__ import annotations

import math
import random

import pytest

from polyt5.evaluation.conformal import (
    ConformalRegressor,
    conformal_quantile,
    split_calibration,
)


def test_the_quantile_uses_the_finite_sample_correction_not_the_plain_quantile():
    """ceil((n+1)(1-alpha))/n, not the plain ceil(n(1-alpha)) quantile.

    n MUST be chosen so the two formulas disagree, or this test cannot detect the
    bug it is named for. At n=20, alpha=0.1: ceil(21*0.9)=ceil(18.9)=19 against
    ceil(20*0.9)=18. A build that drops the (n+1) returns 18.0 and under-covers
    invisibly.

    (n=9 and n=19 both give 9 and 18 under EITHER formula -- an earlier version of
    this test used n=9 and passed against the mutant.)
    """
    residuals = [float(i) for i in range(1, 21)]
    assert conformal_quantile(residuals, alpha=0.1) == 19.0


def test_the_correction_also_discriminates_at_a_different_alpha():
    """n=10, alpha=0.2: ceil(11*0.8)=ceil(8.8)=9 against ceil(10*0.8)=8."""
    residuals = [float(i) for i in range(1, 11)]
    assert conformal_quantile(residuals, alpha=0.2) == 9.0


def test_a_larger_calibration_set_stops_pinning_the_maximum():
    """With n=19 the rank is ceil(20*0.9)=18, the 18th of 19 -- not the max."""
    residuals = [float(i) for i in range(1, 20)]
    assert conformal_quantile(residuals, alpha=0.1) == 18.0


def test_too_few_points_for_the_requested_alpha_is_refused():
    """n=5, alpha=0.1 needs rank 6 > 5. The valid interval is infinite; say so."""
    with pytest.raises(ValueError, match="cannot support alpha"):
        conformal_quantile([1.0, 2.0, 3.0, 4.0, 5.0], alpha=0.1)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_alpha_outside_the_open_unit_interval_is_refused(alpha):
    with pytest.raises(ValueError, match="alpha must be in"):
        conformal_quantile([1.0, 2.0, 3.0], alpha=alpha)


def test_empty_calibration_is_refused():
    with pytest.raises(ValueError, match="empty residual set"):
        conformal_quantile([], alpha=0.1)


def test_coverage_on_exchangeable_data_reaches_the_target():
    """The guarantee, exercised end to end on data that satisfies its one assumption.

    Heavy-tailed noise on purpose: conformal claims validity with NO distributional
    assumption, so a test that only used Gaussians would not distinguish this
    implementation from one that assumed normality and computed 1.645 sigma.
    """
    rng = random.Random(7)

    def noise():
        return rng.gauss(0, 20) if rng.random() < 0.9 else rng.gauss(0, 200)

    cal_t = [rng.uniform(200, 600) for _ in range(2000)]
    cal_p = [t + noise() for t in cal_t]
    val_t = [rng.uniform(200, 600) for _ in range(2000)]
    val_p = [t + noise() for t in val_t]

    model = ConformalRegressor.calibrate(cal_p, cal_t, alpha=0.1)
    report = model.coverage(val_p, val_t)
    assert report.coverage_ci[0] <= 0.90 <= report.coverage_ci[1]


def test_a_tighter_alpha_produces_a_wider_interval():
    rng = random.Random(11)
    t = [rng.uniform(200, 600) for _ in range(1000)]
    p = [x + rng.gauss(0, 25) for x in t]
    assert (
        ConformalRegressor.calibrate(p, t, alpha=0.01).half_width
        > ConformalRegressor.calibrate(p, t, alpha=0.20).half_width
    )


def test_interval_is_centred_on_the_prediction():
    model = ConformalRegressor(half_width=30.0, alpha=0.1, n_calibration=500)
    assert model.interval(412.0) == (382.0, 442.0)


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="differ in length"):
        ConformalRegressor.calibrate([1.0, 2.0], [1.0], alpha=0.1)


def test_coverage_on_an_empty_set_is_refused():
    model = ConformalRegressor(half_width=10.0, alpha=0.1, n_calibration=100)
    with pytest.raises(ValueError, match="empty set"):
        model.coverage([], [])


def test_split_calibration_returns_disjoint_halves_covering_everything():
    cal, val = split_calibration(400, 200, seed=0)
    assert len(cal) == 200 and len(val) == 200
    assert not set(cal) & set(val)
    assert set(cal) | set(val) == set(range(400))


def test_split_calibration_is_deterministic_for_a_seed():
    assert split_calibration(400, 200, seed=3) == split_calibration(400, 200, seed=3)
    assert split_calibration(400, 200, seed=3) != split_calibration(400, 200, seed=4)


def test_coverage_on_the_calibration_set_is_the_tautology_rank_over_n():
    """Why a DISJOINT validation set is required -- and it is not the usual reason.

    Split conformal fits nothing, so reusing calibration data does not overfit the way
    a trained model would. It is worse than that: the calibration-set coverage is
    exactly ceil((n+1)(1-alpha))/n BY CONSTRUCTION, whatever the data, whatever the
    model, even if the predictor is pure noise. It cannot fail, so it measures nothing.

    Only disjoint data can report whether exchangeability actually held. This test
    pins the tautology so nobody mistakes it for evidence.
    """
    rng = random.Random(5)
    n = 500
    t = [rng.uniform(200, 600) for _ in range(n)]
    p = [x + rng.gauss(0, 30) for x in t]
    model = ConformalRegressor.calibrate(p, t, alpha=0.1)
    assert model.coverage(p, t).coverage == math.ceil((n + 1) * 0.9) / n

    garbage = [rng.uniform(-1000, 1000) for _ in range(n)]
    noise_model = ConformalRegressor.calibrate(garbage, t, alpha=0.1)
    assert noise_model.coverage(garbage, t).coverage == math.ceil((n + 1) * 0.9) / n
