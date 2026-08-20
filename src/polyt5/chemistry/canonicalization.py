"""Canonical forms for polymer repeat units, used for deduplication.

Two writings of the same repeat unit (``[At]CCO[At]`` and ``[At]OCC[At]``, say,
or an aromatic versus a kekulized ring) are the same polymer and must collapse
to one entry when counting uniqueness or novelty. RDKit's canonical SMILES is
the collapsing function; the ``[At]`` termini survive canonicalization as
ordinary atoms, which is precisely why polyT5 uses astatine as the placeholder
rather than the ``*`` wildcard.

Nothing here raises: every function returns ``None`` or ``False`` on failure.
"""

from __future__ import annotations

from rdkit import Chem, RDLogger

from .conversion import _mol_from_psmiles, pselfies_to_psmiles, psmiles_to_pselfies

RDLogger.DisableLog("rdApp.*")

__all__ = ["canonical_pselfies", "canonical_psmiles", "is_same_polymer"]

#: Accepted values for the ``kind`` argument of :func:`is_same_polymer`.
_KINDS = ("psmiles", "pselfies")


def canonical_psmiles(psmiles: str) -> str | None:
    """Return the RDKit canonical form of a PSMILES string.

    Star-style termini are normalised to ``[At]`` first, so ``[*]CCO[*]`` and
    ``[At]CCO[At]`` share one canonical form. The ``[At]`` atoms are preserved
    in the output -- they are part of the polymer's identity for dedup purposes.

    Args:
        psmiles: A PSMILES string in either terminus notation.

    Returns:
        The canonical PSMILES, or ``None`` if the string cannot be parsed or
        sanitized by RDKit.

    Examples:
        >>> canonical_psmiles("[At]OCC[At]")
        '[At]CCO[At]'
    """
    mol = _mol_from_psmiles(psmiles)
    if mol is None:
        return None
    try:
        canonical = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None
    return canonical or None


def canonical_pselfies(pselfies: str) -> str | None:
    """Return the canonical form of a PSELFIES string.

    The round trip is decode -> canonicalize the PSMILES -> re-encode, so two
    PSELFIES strings describing the same polymer produce the same output even
    when the model wrote the atoms in a different order. This is the canonical
    form used for deduplication of generated PSELFIES.

    Args:
        pselfies: A PSELFIES string.

    Returns:
        The canonical PSELFIES, or ``None`` if any stage of the round trip
        fails.
    """
    psmiles = pselfies_to_psmiles(pselfies)
    if psmiles is None:
        return None
    canonical = canonical_psmiles(psmiles)
    if canonical is None:
        return None
    return psmiles_to_pselfies(canonical)


def is_same_polymer(a: str, b: str, *, kind: str = "psmiles") -> bool:
    """Test whether two strings describe the same polymer repeat unit.

    Comparison is made on canonical forms, so equivalent writings compare equal.

    Args:
        a: First string.
        b: Second string.
        kind: ``"psmiles"`` (default) or ``"pselfies"``, selecting how the two
            strings are interpreted.

    Returns:
        ``True`` if both strings canonicalize successfully to the same polymer.
        If either string fails to canonicalize the result is ``False`` -- two
        unparseable strings are not "the same polymer", they are both simply
        not polymers.

    Raises:
        ValueError: If ``kind`` is not one of ``"psmiles"`` or ``"pselfies"``.
            This is a caller programming error, not adversarial data, so it is
            surfaced rather than swallowed.
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS!r}, got {kind!r}")

    canonicalize = canonical_psmiles if kind == "psmiles" else canonical_pselfies
    canonical_a = canonicalize(a)
    if canonical_a is None:
        return False
    canonical_b = canonicalize(b)
    if canonical_b is None:
        return False
    return canonical_a == canonical_b
