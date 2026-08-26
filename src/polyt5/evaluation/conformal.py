"""Split-conformal prediction intervals (Phase 4 Group B).

NOT part of the published polyT5 method -- see
``docs/superpowers/specs/2026-08-25-phase4-group-b-conformal-design.md``.

Turns a point prediction into an interval with a distribution-free, finite-sample
coverage guarantee::

    P(y in C(x)) >= 1 - alpha

The guarantee needs exchangeability between calibration and test data and NOTHING
else -- no Gaussian residuals, no well-specified model, no assumption that the
predictor is any good. That is the whole appeal: it is honest about a model we
have measured to be imperfect.

This module takes NUMBERS, not a model, so it never imports torch and is testable
without a GPU. Wrap any predictor by feeding it that predictor's outputs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["ConformalRegressor", "CoverageReport", "conformal_quantile"]


def conformal_quantile(residuals: Sequence[float], alpha: float) -> float:
    """The split-conformal quantile of ``residuals`` at miscoverage ``alpha``.

    Uses the ``ceil((n + 1) * (1 - alpha)) / n`` empirical quantile, NOT the plain
    ``1 - alpha`` quantile. The ``(n + 1)`` is the finite-sample correction that makes
    the coverage guarantee hold at small ``n``; the plain quantile under-covers, and
    the shortfall is invisible without a coverage test.

    Args:
        residuals: Absolute errors on a calibration set the model never saw.
        alpha: Miscoverage rate, e.g. 0.1 for 90% coverage. Must be in (0, 1).

    Returns:
        The half-width to add and subtract from a point prediction.

    Raises:
        ValueError: If ``residuals`` is empty, if ``alpha`` is outside (0, 1), or if
            ``n`` is too small for this ``alpha`` -- that is, if the required rank
            exceeds ``n``, where the honest answer is an infinite interval and the
            useful answer is "collect more calibration data".
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    values = sorted(float(r) for r in residuals)
    n = len(values)
    if n == 0:
        raise ValueError("cannot calibrate on an empty residual set")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        raise ValueError(
            f"{n} calibration points cannot support alpha={alpha}: the conformal rank is "
            f"{rank}, above n, so the only valid interval is infinite. Need at least "
            f"{math.ceil(1.0 / alpha) - 1} points."
        )
    return values[rank - 1]


@dataclass(frozen=True)
class CoverageReport:
    """Measured coverage of an interval on data that never entered calibration."""

    n: int
    covered: int
    coverage: float
    target_coverage: float
    mean_width: float
    #: Wald 95% interval on the coverage estimate. Reported because a small
    #: validation set cannot distinguish 0.90 from 0.85, and a bare point
    #: estimate invites the reader to believe it can.
    coverage_ci: tuple[float, float]


@dataclass(frozen=True)
class ConformalRegressor:
    """A calibrated constant-width interval.

    Deliberately constant-width. Normalized conformal would scale each interval by a
    per-input uncertainty estimate, but ``docs/instrument_audit.md`` measured
    ``corr(sigma, |error|) = 0.15`` for our ensemble disagreement -- sigma explains
    about 2% of error variance, so dividing by it injects noise. An honestly
    calibrated constant width beats a variable width built on an uninformative sigma.
    """

    half_width: float
    alpha: float
    n_calibration: int

    @classmethod
    def calibrate(
        cls,
        predictions: Sequence[float],
        targets: Sequence[float],
        alpha: float = 0.1,
    ) -> ConformalRegressor:
        """Fit on a calibration set the model neither trained on nor selected against."""
        if len(predictions) != len(targets):
            raise ValueError(
                f"predictions and targets differ in length: {len(predictions)} vs {len(targets)}"
            )
        residuals = [abs(float(p) - float(t)) for p, t in zip(predictions, targets, strict=True)]
        return cls(
            half_width=conformal_quantile(residuals, alpha),
            alpha=alpha,
            n_calibration=len(residuals),
        )

    def interval(self, prediction: float) -> tuple[float, float]:
        """The interval around one point prediction."""
        return (float(prediction) - self.half_width, float(prediction) + self.half_width)

    def coverage(
        self, predictions: Sequence[float], targets: Sequence[float]
    ) -> CoverageReport:
        """Measure realised coverage on held-out data.

        Raises:
            ValueError: If lengths differ or the set is empty.

        Note:
            On the calibration set itself this returns exactly ``rank / n`` by
            construction -- deterministic, not inflated. Split conformal does not fit
            anything, so reusing calibration data does not overfit the way a trained
            model would. The reason to hold out a separate validation set is that
            ``rank / n`` is a tautology and measures nothing: only disjoint data can
            tell you whether the exchangeability assumption actually held.
        """
        if len(predictions) != len(targets):
            raise ValueError(
                f"predictions and targets differ in length: {len(predictions)} vs {len(targets)}"
            )
        if not predictions:
            raise ValueError("cannot measure coverage on an empty set")
        hits = sum(
            1
            for p, t in zip(predictions, targets, strict=True)
            if self.interval(p)[0] <= float(t) <= self.interval(p)[1]
        )
        n = len(predictions)
        rate = hits / n
        se = math.sqrt(max(rate * (1.0 - rate), 0.0) / n)
        return CoverageReport(
            n=n,
            covered=hits,
            coverage=rate,
            target_coverage=1.0 - self.alpha,
            mean_width=2.0 * self.half_width,
            coverage_ci=(max(0.0, rate - 1.96 * se), min(1.0, rate + 1.96 * se)),
        )


def split_calibration(
    n: int, n_calibration: int, seed: int
) -> tuple[list[int], list[int]]:
    """Split ``n`` held-out indices into disjoint calibration and validation halves.

    Exists so that calibrating and validating on the same data takes a deliberate act
    rather than an oversight: reusing the calibration set inflates measured coverage
    toward 1 and would make the interval look better than it is.
    """
    import random

    if not 0 < n_calibration < n:
        raise ValueError(f"n_calibration must be in (0, {n}), got {n_calibration}")
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return sorted(idx[:n_calibration]), sorted(idx[n_calibration:])
