"""Deterministic enumeration of alternative writings of one polymer.

One polymer has many valid SMILES writings, and therefore many valid PSELFIES
writings. Register entry A-05 established that canonicalisation collapses them
back to exactly one form, deterministically across processes -- which is what
makes them safe to train on: the model learns Tg is a property of the molecule,
not of the string.

RDKit note, verified on this machine (rdkit 2026.03.5):
``Chem.MolToRandomSmilesVect(mol, n, randomSeed=0)`` is NOT deterministic --
``0`` is RDKit's "choose a seed for me" sentinel. Every call here passes
``randomSeed=seed + 1`` so a caller's ``seed=0`` still reproduces.

Torch-free: rdkit only, like the rest of :mod:`polyt5.chemistry`. Nothing here
raises on adversarial input; an unparseable string yields an empty list.
"""

from __future__ import annotations

from rdkit import Chem, RDLogger

from .canonicalization import canonical_psmiles
from .conversion import _mol_from_psmiles, psmiles_to_pselfies

RDLogger.DisableLog("rdApp.*")

__all__ = ["enumerate_pselfies_writings"]

#: How many random SMILES to request per writing wanted. RDKit returns
#: duplicates freely, and some writings fail the SELFIES round trip, so the
#: request is oversubscribed rather than looped.
_OVERSAMPLE = 6


def enumerate_pselfies_writings(psmiles: str, n_writings: int, *, seed: int = 0) -> list[str]:
    """Return up to ``n_writings`` distinct PSELFIES writings of one polymer.

    Args:
        psmiles: A PSMILES string in either terminus notation.
        n_writings: Maximum number of writings to return (>= 1).
        seed: Reproducibility seed. The same ``(psmiles, n_writings, seed)``
            always yields the same list, in the same order.

    Returns:
        Distinct PSELFIES strings, the canonical writing first, each of which
        canonicalises back to the same polymer as ``psmiles``. An empty list
        when ``psmiles`` cannot be parsed. Never raises on bad input.

    Raises:
        ValueError: If ``n_writings`` is below 1. That is a caller bug, not
            adversarial data.
    """
    if n_writings < 1:
        raise ValueError(f"n_writings must be >= 1, got {n_writings}")

    mol = _mol_from_psmiles(psmiles)
    if mol is None:
        return []
    reference = canonical_psmiles(psmiles)
    if reference is None:
        return []

    candidates: list[str] = [reference]
    try:
        candidates.extend(
            Chem.MolToRandomSmilesVect(mol, n_writings * _OVERSAMPLE, randomSeed=seed + 1)
        )
    except Exception:  # RDKit failures are data problems, not caller bugs
        pass

    writings: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if canonical_psmiles(candidate) != reference:
            continue
        pselfies = psmiles_to_pselfies(candidate)
        if pselfies is None or pselfies in seen:
            continue
        seen.add(pselfies)
        writings.append(pselfies)
        if len(writings) >= n_writings:
            break
    return writings
