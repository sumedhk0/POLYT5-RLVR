"""Confidence-weighted Tg reward.

The Tg reward is the only term backed by a model rather than a computed fact,
so it is the only term a policy can farm. Our five predictors disagree by
16.7 K on average and up to 45.2 K, so exploitable regions demonstrably exist.
The reward is therefore scaled by ensemble agreement.

This discounts the *accuracy claim*, never novelty: validity, novelty and
diversity rewards carry no similarity constraint, so a novel valid polymer
keeps full credit on those axes. See spec section 4.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from polyt5.rewards.base import RewardResult

#: Observed mean inter-model disagreement on real generations, in Kelvin.
DEFAULT_SIGMA0 = 17.0


@dataclass(frozen=True)
class TgRewardConfig:
    """Knobs for the Tg term.

    Attributes:
        tolerance: Error at which closeness reaches zero, in Kelvin.
        sigma0: Disagreement scale; the confidence weight is 0.5 at this value.
    """

    tolerance: float = 100.0
    sigma0: float = DEFAULT_SIGMA0

    def __post_init__(self) -> None:
        if self.tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")
        if self.sigma0 <= 0:
            raise ValueError(f"sigma0 must be positive, got {self.sigma0}")


#: Shared default config. Safe to reuse (unlike RewardResult): TgRewardConfig
#: is frozen and holds only immutable floats, so no caller can corrupt it by
#: mutating a field in place.
_DEFAULT_CONFIG = TgRewardConfig()


def tg_reward(
    predicted: float,
    std: float,
    target: float,
    *,
    config: TgRewardConfig = _DEFAULT_CONFIG,
) -> RewardResult:
    """Score how close a prediction is to target, discounted by disagreement.

    Args:
        predicted: Ensemble mean prediction, Kelvin.
        std: Ensemble standard deviation, Kelvin.
        target: Requested Tg, Kelvin.
        config: Tolerance and disagreement scale.

    Returns:
        RewardResult whose ``components`` carry the unweighted ``closeness``
        and the ``confidence`` factor separately, so the gate's effect on the
        learning signal can be measured rather than assumed.
    """
    if not (math.isfinite(predicted) and math.isfinite(std) and math.isfinite(target)):
        return RewardResult(0.0, {"closeness": 0.0, "confidence": 0.0}, True, "non_finite")

    closeness = max(0.0, 1.0 - abs(predicted - target) / config.tolerance)
    confidence = 1.0 / (1.0 + max(0.0, std) / config.sigma0)
    return RewardResult(
        value=closeness * confidence,
        components={
            "closeness": closeness,
            "confidence": confidence,
            "abs_error": abs(predicted - target),
            "std": std,
        },
    )
