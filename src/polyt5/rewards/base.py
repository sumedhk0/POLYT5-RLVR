"""Reward primitives shared by every arm.

This package is deliberately torch-free: reward workers score candidates on
CPU without a model in the process. A subprocess test enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RewardResult:
    """One candidate's reward, with its parts kept for logging.

    Attributes:
        value: The scalar reward actually used by GRPO.
        components: Named sub-scores, for diagnosing which term drove the value.
        gated: True when the validity gate zeroed this candidate.
        reason: Short machine-readable tag when gated, else None.
    """

    value: float
    components: dict[str, float] = field(default_factory=dict)
    gated: bool = False
    reason: str | None = None
