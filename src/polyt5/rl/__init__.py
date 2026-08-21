"""GRPO/RLVR extension. Phase 3 - not part of the published polyT5 method."""

from __future__ import annotations

from polyt5.rl.advantages import group_advantages
from polyt5.rl.drift import DriftMonitor
from polyt5.rl.grpo import GRPOConfig, grpo_loss, k3_kl
from polyt5.rl.reference_policy import ReferencePolicy
from polyt5.rl.rollout import ROLLOUT_CHUNK_SIZE, RolloutBatch, sample_groups
from polyt5.rl.trainer import GRPOTrainer, GRPOTrainerConfig

__all__ = [
    "GRPOConfig",
    "GRPOTrainer",
    "GRPOTrainerConfig",
    "ROLLOUT_CHUNK_SIZE",
    "DriftMonitor",
    "ReferencePolicy",
    "RolloutBatch",
    "grpo_loss",
    "group_advantages",
    "k3_kl",
    "sample_groups",
]
