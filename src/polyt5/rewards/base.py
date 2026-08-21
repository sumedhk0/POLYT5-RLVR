"""Reward primitives shared by every arm.

This package is deliberately torch-free: reward workers score candidates on
CPU without a model in the process. A subprocess test enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RewardResult:
    """One candidate's reward, with its parts kept for logging.

    ``frozen=True`` blocks reassigning ``value``, ``components``, ``gated``, or
    ``reason`` on an existing instance, but it does not deep-freeze
    ``components`` itself: that dict is still mutable in place. Callers must
    never write into a ``RewardResult.components`` they did not just create,
    and constructors must never return a module-level shared instance for
    this reason - a later in-place edit would corrupt every other holder of
    the same object.

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
