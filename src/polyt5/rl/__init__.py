"""GRPO/RLVR extension. Phase 3 - not part of the published polyT5 method."""

from __future__ import annotations

from polyt5.rl.advantages import group_advantages
from polyt5.rl.grpo import GRPOConfig, grpo_loss, k3_kl

__all__ = ["GRPOConfig", "grpo_loss", "group_advantages", "k3_kl"]
