"""Synthetic accessibility term, normalised into [0, 1].

RDKit's SA score runs 1 (easy) to 10 (hard). The paper reports known polymers
mostly in the 2-3 range and treats 6 as the threshold for awkward candidates.
"""

from __future__ import annotations

from polyt5.rewards.base import RewardResult


def sa_reward(sa_score: float | None, *, sa_max: float = 6.0) -> RewardResult:
    """Score synthesisability; 1.0 at SA=1, 0.0 at or beyond ``sa_max``.

    Args:
        sa_score: RDKit SA score, or None when unavailable on this machine.
        sa_max: Score at which the reward reaches zero.

    Returns:
        RewardResult with an ``sa`` component. None input yields 0.0 and is
        NOT treated as a pass, so a missing scorer cannot inflate rewards.
    """
    if sa_score is None:
        return RewardResult(0.0, {"sa": 0.0})
    value = max(0.0, min(1.0, (sa_max - sa_score) / (sa_max - 1.0)))
    return RewardResult(value, {"sa": value, "sa_score": sa_score})
