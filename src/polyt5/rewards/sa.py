"""SA (synthetic accessibility) reward term, shared by every arm that checks it.

[PRE-REGISTRATION NOTE, dated 2026-08-23] This module is new. It exists because
Tg was dropped from every RLVR reward (see ``composite.py``'s module
docstring and ``artifacts/baseline/frozen_baseline.json``'s amendment of the
same date): a generated polymer has no experimental Tg and never will, so a
Tg reward is a learned model's opinion, not a computed fact, and arms built on
it are not RLVR. SA is different -- RDKit's ``sascorer`` heuristic is a
deterministic function of the candidate's own structure, computed the same
way regardless of who is asking, so a threshold on it is verifiable in
exactly the sense a Tg claim is not.

An identically-named public symbol existed once before and was removed as
dead code (no arm called it -- :class:`~polyt5.rewards.composite.ConstraintArm`
read the raw SA score and applied its own threshold instead, so wiring the old
symbol in would have silently changed C4's reward at the boundary). This is
not that symbol restored unchanged: it is wired into three arms now
(:class:`~polyt5.rewards.composite.SynthesisabilityArm`,
:class:`~polyt5.rewards.composite.CompositeArm`,
:class:`~polyt5.rewards.composite.ConstraintArm`) precisely so the threshold
check lives in ONE place instead of being duplicated across them.
"""

from __future__ import annotations

from polyt5.chemistry.metrics import synthetic_accessibility
from polyt5.rewards.base import RewardResult


def sa_pass(sa_score: float | None, *, sa_max: float) -> bool:
    """Whether an RDKit SA score clears the synthesisability threshold.

    Args:
        sa_score: The RDKit SA score (1 easy - 10 hard), or ``None`` when the
            structure could not be parsed or the RDKit Contrib scorer is
            unavailable (:func:`~polyt5.chemistry.metrics.synthetic_accessibility`).
        sa_max: Maximum acceptable score.

    Returns:
        ``False`` when ``sa_score`` is ``None`` -- a missing score must never
        be read as passing -- else ``sa_score <= sa_max``.
    """
    return sa_score is not None and sa_score <= sa_max


def sa_reward(canonical_psmiles: str | None, *, sa_max: float = 6.0) -> RewardResult:
    """Score synthesisability alone: a boolean SA-threshold gate.

    Verifiable: the SA score is a deterministic RDKit heuristic computed on
    the candidate's own structure, never a model's prediction about it and
    never conditioned on a target the candidate cannot be checked against.

    Args:
        canonical_psmiles: Canonical PSMILES, or ``None`` when decoding
            failed -- scored as a fail, mirroring
            :func:`~polyt5.rewards.novelty.novelty_reward`'s treatment of a
            missing structure.
        sa_max: Maximum acceptable SA score.

    Returns:
        RewardResult whose value is ``1.0`` iff the SA score is defined and
        ``<= sa_max``, else ``0.0``. ``components`` always carries
        ``synthesisable``, and additionally ``sa_score`` whenever the score
        could be computed at all, so a ``None`` case (unparseable structure or
        the RDKit Contrib scorer unavailable) is distinguishable from a
        genuinely bad score in step logs.
    """
    sa_score = synthetic_accessibility(canonical_psmiles) if canonical_psmiles else None
    passed = sa_pass(sa_score, sa_max=sa_max)
    components = {"synthesisable": float(passed)}
    if sa_score is not None:
        components["sa_score"] = float(sa_score)
    return RewardResult(float(passed), components)
