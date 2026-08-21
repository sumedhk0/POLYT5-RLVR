"""Confidence-weighted Tg reward.

The Tg reward is the only term backed by a model rather than a computed fact,
so it is the only term a policy can farm. Our five predictors disagree by
16.7 K on average and up to 45.2 K, so exploitable regions demonstrably exist.
The reward is therefore scaled by ensemble agreement.

This discounts the *accuracy claim*, never novelty: validity, novelty and
diversity rewards carry no similarity constraint, so a novel valid polymer
keeps full credit on those axes. See spec section 4.2.

Two DIFFERENT failures both used to arrive here as ``std == 0.0``
---------------------------------------------------------------
:meth:`polyt5.inference.predictor.EnsemblePropertyPredictor.predict_with_uncertainty`
drops any member that returned no parseable number from that candidate's mean
and std. ``std`` is therefore the population standard deviation *over the
members that actually answered*, which is ``0.0`` in two completely different
situations:

* every member answered and they all agreed -- genuine, maximal confidence;
* exactly ONE member answered -- there is nothing to disagree with, so the
  spread is undefined, not zero.

Weighting both by ``1 / (1 + std / sigma0) == 1.0`` inverted the whole point of
the gate: a candidate three quarters of the ensemble could not parse scored
**1.0000** while one all four agreed on (spread 4 K) scored **0.8095**, i.e. an
*ascending* gradient pointing straight at chemistry that breaks the reward
models' decoders. The third element of the prediction triple,
``n_contributing_members``, is what separates the two cases, so it is now a
required argument rather than something each arm discarded:

* ``n_contributing == 0`` -- nothing could score the candidate. Reward ``0.0``.
* ``n_contributing == 1`` while the predictor has more than one member -- the
  spread is UNDEFINED. :attr:`TgRewardConfig.sigma_unknown` (45.2 K, our
  maximum observed inter-model disagreement) is substituted for it, never
  ``0.0``.
* otherwise -- the reported ``std`` is a real spread over >= 2 members.

and the whole weight is scaled by the fraction of the ensemble that answered::

    confidence = (n_contributing / n_total) * 1 / (1 + sigma_effective / sigma0)

``n_total == 1`` (a genuine single-model predictor, e.g. ``compare_arms``'s
held-out auditor, which reports ``(mean, 0.0, 1)``) is deliberately NOT the
undefined-spread case: there is no ensemble whose members could have failed, so
coverage is ``1/1`` and ``sigma_effective`` is the reported ``0.0``, leaving the
auditor-side score as unweighted closeness exactly as Ruling C intends. The
distinction "1 of 1" vs "1 of 4" is carried entirely by ``n_total``, which is
why arms must declare their predictor's size (``_BaseArm.ensemble_size``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from polyt5.rewards.base import RewardResult

#: Observed mean inter-model disagreement on real generations, in Kelvin.
DEFAULT_SIGMA0 = 17.0

#: Observed MAXIMUM inter-model disagreement, in Kelvin. Substituted for the
#: spread when exactly one member of a multi-member ensemble could score a
#: candidate: the true spread is unknown there, and the pessimistic value is
#: the only substitution that cannot inflate a reward.
DEFAULT_SIGMA_UNKNOWN = 45.2

#: Minimum fraction of the ensemble that must have scored a candidate for an
#: arm WITHOUT a continuous confidence weight (C4, whose reward is a boolean
#: conjunction by spec) to treat the ensemble mean as a consensus at all.
DEFAULT_MIN_COVERAGE = 0.5


@dataclass(frozen=True)
class TgRewardConfig:
    """Knobs for the Tg term.

    Attributes:
        tolerance: Error at which closeness reaches zero, in Kelvin.
        sigma0: Disagreement scale; the confidence weight is 0.5 at this value.
        sigma_unknown: Spread substituted when the spread is undefined -- one
            contributing member out of several. Our maximum observed
            disagreement (45.2 K), so an unscoreable candidate is treated as
            the worst disagreement we have ever measured rather than the best.
        min_coverage: Minimum ``n_contributing / n_total`` for
            :class:`~polyt5.rewards.composite.ConstraintArm` to accept the
            ensemble mean as a consensus. Unused by the continuous
            confidence weight, which scales by coverage instead of gating on it.
    """

    tolerance: float = 100.0
    sigma0: float = DEFAULT_SIGMA0
    sigma_unknown: float = DEFAULT_SIGMA_UNKNOWN
    min_coverage: float = DEFAULT_MIN_COVERAGE

    def __post_init__(self) -> None:
        if self.tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")
        if self.sigma0 <= 0:
            raise ValueError(f"sigma0 must be positive, got {self.sigma0}")
        if self.sigma_unknown < 0:
            raise ValueError(f"sigma_unknown must be >= 0, got {self.sigma_unknown}")
        if not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError(f"min_coverage must be in [0, 1], got {self.min_coverage}")


#: Shared default config. Safe to reuse (unlike RewardResult): TgRewardConfig
#: is frozen and holds only immutable floats, so no caller can corrupt it by
#: mutating a field in place.
_DEFAULT_CONFIG = TgRewardConfig()


def effective_sigma(
    std: float, n_contributing: int, n_total: int, *, config: TgRewardConfig = _DEFAULT_CONFIG
) -> float:
    """The spread to weight by, with the undefined case named explicitly.

    Args:
        std: The spread the predictor reported over its contributing members.
        n_contributing: How many members actually returned a number.
        n_total: How many members the predictor has in total.
        config: Supplies ``sigma_unknown``.

    Returns:
        ``config.sigma_unknown`` when the spread is undefined (exactly one of
        several members answered), and the reported ``std`` otherwise. A
        single-member predictor (``n_total == 1``) is not the undefined case --
        see the module docstring.
    """
    if n_contributing == 1 and n_total > 1:
        return config.sigma_unknown
    return max(0.0, std)


def tg_reward(
    predicted: float,
    std: float,
    target: float,
    *,
    n_contributing: int,
    n_total: int,
    config: TgRewardConfig = _DEFAULT_CONFIG,
) -> RewardResult:
    """Score how close a prediction is to target, discounted by disagreement.

    Args:
        predicted: Ensemble mean prediction over the CONTRIBUTING members,
            Kelvin.
        std: Ensemble standard deviation over the contributing members, Kelvin.
        target: Requested Tg, Kelvin.
        n_contributing: How many ensemble members actually produced a number
            for this candidate -- the third element of the predictor's
            ``(mean, std, n_contributing_members)`` triple. Required, not
            optional: discarding it is exactly the defect this argument exists
            to close (see the module docstring).
        n_total: How many members the predictor that produced ``std`` has.
            ``1`` for a genuine single-model predictor.
        config: Tolerance, disagreement scale, and the substituted spread.

    Returns:
        RewardResult whose ``components`` carry the unweighted ``closeness``
        and the ``confidence`` factor separately, so the gate's effect on the
        learning signal can be measured rather than assumed, plus
        ``n_contributing`` / ``coverage`` / ``sigma_effective`` so a partial
        ensemble is visible in the step logs rather than inferred.

    Raises:
        ValueError: If ``n_total < 1`` or ``n_contributing`` is negative or
            exceeds ``n_total``. The last case means the arm's declared
            ``ensemble_size`` disagrees with the predictor it is actually being
            fed, which would silently understate the coverage penalty; a run
            must die at step 0 rather than train against it.
    """
    if n_total < 1:
        raise ValueError(f"n_total must be >= 1, got {n_total}")
    if not 0 <= n_contributing <= n_total:
        raise ValueError(
            f"n_contributing ({n_contributing}) must be in [0, n_total] (n_total={n_total}): "
            "the arm's declared ensemble_size does not match the predictor supplying these "
            "predictions, so the coverage penalty would be computed against the wrong "
            "denominator."
        )

    if n_contributing == 0:
        return RewardResult(
            0.0,
            {"closeness": 0.0, "confidence": 0.0, "n_contributing": 0.0, "coverage": 0.0},
            True,
            "no_contributing_members",
        )

    if not (math.isfinite(predicted) and math.isfinite(std) and math.isfinite(target)):
        return RewardResult(
            0.0,
            {"closeness": 0.0, "confidence": 0.0,
             "n_contributing": float(n_contributing),
             "coverage": n_contributing / n_total},
            True,
            "non_finite",
        )

    closeness = max(0.0, 1.0 - abs(predicted - target) / config.tolerance)
    sigma = effective_sigma(std, n_contributing, n_total, config=config)
    coverage = n_contributing / n_total
    confidence = coverage * (1.0 / (1.0 + sigma / config.sigma0))
    return RewardResult(
        value=closeness * confidence,
        components={
            "closeness": closeness,
            "confidence": confidence,
            "abs_error": abs(predicted - target),
            "std": std,
            "sigma_effective": sigma,
            "n_contributing": float(n_contributing),
            "coverage": coverage,
        },
    )
