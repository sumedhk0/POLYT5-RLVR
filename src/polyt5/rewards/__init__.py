"""Verifiable reward components for the GRPO/RLVR extension.

[PRE-REGISTRATION NOTE, dated 2026-08-23] Tg was dropped from every RLVR
reward on this date -- see :mod:`polyt5.rewards.composite`'s module
docstring. ``sa_reward`` (:mod:`polyt5.rewards.sa`) is part of that change: an
identically-named symbol existed once before as dead code (no arm called it)
and was removed; it is back now, wired into three arms
(:class:`~polyt5.rewards.composite.SynthesisabilityArm`,
:class:`~polyt5.rewards.composite.CompositeArm`,
:class:`~polyt5.rewards.composite.ConstraintArm`) so the SA-threshold check
lives in one place instead of being duplicated across them.
"""

from __future__ import annotations

from polyt5.rewards.base import RewardResult
from polyt5.rewards.composite import (
    DEFAULT_COMPOSITE_WEIGHTS,
    AccuracyArm,
    ArmReward,
    CompositeArm,
    ConstraintArm,
    ControlArm,
    NoveltyArm,
    SynthesisabilityArm,
    ValidityArm,
    build_arm,
)
from polyt5.rewards.constraints import constraint_reward
from polyt5.rewards.novelty import novelty_reward
from polyt5.rewards.sa import sa_pass, sa_reward
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
    "ControlArm",
    "NoveltyArm",
    "RewardResult",
    "SynthesisabilityArm",
    "TgRewardConfig",
    "ValidityArm",
    "build_arm",
    "constraint_reward",
    "effective_sigma",
    "novelty_reward",
    "sa_pass",
    "sa_reward",
    "tg_reward",
    "validity_gate",
]
