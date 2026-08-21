"""The validity gate: a structure that is not a polymer earns nothing.

The paper's PV rule is structural - exactly two astatine termini, each with
valency one - and it is checked here rather than being one term among many,
so that invalid chemistry cannot accumulate partial credit on other axes.
"""

from __future__ import annotations

from polyt5.chemistry.conversion import pselfies_to_psmiles
from polyt5.chemistry.validity import validate_pselfies
from polyt5.evaluation import has_valid_termini
from polyt5.rewards.base import RewardResult

_PASS = RewardResult(value=1.0, components={"validity": 1.0}, gated=False)


def validity_gate(pselfies: str, *, expected_termini: int = 2) -> RewardResult:
    """Return a passing result, or a gated zero with a reason.

    Args:
        pselfies: Candidate PSELFIES string, straight from the model.
        expected_termini: Required number of ``[At]`` atoms.

    Returns:
        RewardResult with value 1.0 when the candidate is a well-formed
        polymer, else 0.0 with ``gated=True``. Never raises.
    """
    try:
        verdict = validate_pselfies(pselfies, expected_termini=expected_termini)
        if not verdict.valid:
            return RewardResult(0.0, {"validity": 0.0}, True, verdict.reason or "invalid")
        if not verdict.correct_termini:
            return RewardResult(0.0, {"validity": 0.0}, True, "wrong_termini_count")
        psmiles = pselfies_to_psmiles(pselfies)
        if psmiles is None or not has_valid_termini(psmiles, expected=expected_termini):
            return RewardResult(0.0, {"validity": 0.0}, True, "terminus_valency")
        return _PASS
    except Exception:  # a reward must never crash a training step
        return RewardResult(0.0, {"validity": 0.0}, True, "exception")
