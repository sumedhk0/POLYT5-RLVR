"""Verifiable reward components for the GRPO/RLVR extension.

Note:
    There is deliberately no ``sa_reward`` here. A normalised synthetic-
    accessibility term existed as a public symbol that no arm ever called:
    :class:`~polyt5.rewards.composite.ConstraintArm` reads the raw RDKit SA
    score and :func:`~polyt5.rewards.constraints.constraint_reward` applies the
    ``sa <= sa_max`` threshold itself. Wiring it in would have changed C4's
    reward at the threshold (``sa_reward`` reaches 0.0 *at* ``sa_max`` where
    ``constraint_reward`` still passes), which is a spec change and not a
    review fix, so the unused symbol was removed instead. Restore it from
    history if a continuous SA term is ever actually wanted.
"""

from __future__ import annotations

from polyt5.rewards.base import RewardResult
from polyt5.rewards.composite import (
    DEFAULT_COMPOSITE_WEIGHTS,
    AccuracyArm,
    ArmReward,
    CompositeArm,
    ConstraintArm,
    ValidityArm,
    build_arm,
)
from polyt5.rewards.constraints import constraint_reward
from polyt5.rewards.novelty import novelty_reward
from polyt5.rewards.tg import (
    DEFAULT_MIN_COVERAGE,
    DEFAULT_SIGMA0,
    DEFAULT_SIGMA_UNKNOWN,
    TgRewardConfig,
    effective_sigma,
    tg_reward,
)
from polyt5.rewards.validity import validity_gate

__all__ = [
    "DEFAULT_COMPOSITE_WEIGHTS",
    "DEFAULT_MIN_COVERAGE",
    "DEFAULT_SIGMA0",
    "DEFAULT_SIGMA_UNKNOWN",
    "AccuracyArm",
    "ArmReward",
    "CompositeArm",
    "ConstraintArm",
    "RewardResult",
    "TgRewardConfig",
    "ValidityArm",
    "build_arm",
    "constraint_reward",
    "effective_sigma",
    "novelty_reward",
    "tg_reward",
    "validity_gate",
]
