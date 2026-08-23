"""Arm C4: a conjunction, so partial satisfaction earns nothing.

[PRE-REGISTRATION NOTE, dated 2026-08-23] ``constraint_reward`` used to be a
FOUR-way conjunction (Tg window AND synthesisable AND novel AND
ensemble-backed). Tg was dropped from every RLVR reward on this date -- see
:mod:`polyt5.rewards.composite`'s module docstring and
``artifacts/baseline/frozen_baseline.json``'s amendment of the same date --
which removes both the ``in_window`` clause and the ``ensemble_backed``
clause (the latter existed only to stop a single ensemble member's guess
being read as a consensus for the Tg clause; with no Tg clause there is
nothing left for it to guard). This arm has NEVER been trained: the change is
a pre-registration made before any observation of C4's results, not a
criterion chosen after seeing them.
"""

from __future__ import annotations

from polyt5.rewards.base import RewardResult
from polyt5.rewards.sa import sa_pass


def constraint_reward(
    sa_score: float | None,
    novel: bool,
    *,
    sa_max: float = 6.0,
) -> RewardResult:
    """Return 1.0 only when every condition holds simultaneously.

    Args:
        sa_score: RDKit SA score, or None.
        novel: Whether the candidate is absent from the reference corpus.
        sa_max: Maximum acceptable SA score.

    Returns:
        RewardResult whose components record each condition separately, so a
        failure can be attributed to the condition that caused it.
    """
    synthesisable = sa_pass(sa_score, sa_max=sa_max)
    satisfied = bool(synthesisable and novel)
    return RewardResult(
        float(satisfied),
        {"synthesisable": float(synthesisable), "novel": float(novel)},
    )
