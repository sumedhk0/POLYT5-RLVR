"""Arm C4: a conjunction, so partial satisfaction earns nothing."""

from __future__ import annotations

from polyt5.rewards.base import RewardResult


def constraint_reward(
    abs_error: float,
    sa_score: float | None,
    novel: bool,
    *,
    tolerance: float = 50.0,
    sa_max: float = 6.0,
) -> RewardResult:
    """Return 1.0 only when every condition holds simultaneously.

    Args:
        abs_error: |predicted Tg - target Tg|, Kelvin.
        sa_score: RDKit SA score, or None.
        novel: Whether the candidate is absent from the reference corpus.
        tolerance: Tg window half-width, Kelvin.
        sa_max: Maximum acceptable SA score.

    Returns:
        RewardResult whose components record each condition separately, so a
        failure can be attributed to the condition that caused it.
    """
    in_window = abs_error <= tolerance
    synthesisable = sa_score is not None and sa_score <= sa_max
    satisfied = bool(in_window and synthesisable and novel)
    return RewardResult(
        float(satisfied),
        {"in_window": float(in_window), "synthesisable": float(synthesisable),
         "novel": float(novel)},
    )
