"""Verifiable reward components for the GRPO/RLVR extension."""

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
from polyt5.rewards.sa import sa_reward
from polyt5.rewards.tg import DEFAULT_SIGMA0, TgRewardConfig, tg_reward
from polyt5.rewards.validity import validity_gate

__all__ = [
    "DEFAULT_COMPOSITE_WEIGHTS",
    "DEFAULT_SIGMA0",
    "AccuracyArm",
    "ArmReward",
    "CompositeArm",
    "ConstraintArm",
    "RewardResult",
    "TgRewardConfig",
    "ValidityArm",
    "build_arm",
    "constraint_reward",
    "novelty_reward",
    "sa_reward",
    "tg_reward",
    "validity_gate",
]
