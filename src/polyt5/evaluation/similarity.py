"""ECFP6 fingerprints and Tanimoto similarity, with the paper's loop closure.

Sahu et al. measure structural similarity between polymers with "ECFP6
2048-bit fingerprints as implemented in RDKit. To eliminate terminal effects and
better approximate infinite polymer chains, the two ends of each repeating unit
were connected to form a loop prior to fingerprint calculation."

That loop closure is the whole point: without it, the ``[At]`` caps dominate the
circular environments at the chain ends and two chemically similar repeat units
look artificially different. :func:`loop_closed_mol` implements it by deleting
the two terminus atoms and bonding their former neighbours to each other.

Why ``rdMolDescriptors`` and not the modern generator API
---------------------------------------------------------
The current RDKit way to build ECFP is ``rdFingerprintGenerator``. On this
machine that extension module cannot be loaded at all::

    ImportError: DLL load failed while importing rdFingerprintGenerator:
    An Application Control policy has blocked this file.

which also takes out ``AllChem.GetMorganFingerprintAsBitVect`` (a thin wrapper
over it). The older ``rdMolDescriptors.GetMorganFingerprintAsBitVect`` entry
point loads fine and computes the identical fingerprint, so this module calls it
directly and silences its deprecation warning *locally*, at the call site, rather
than installing a global warning filter that would hide the same warning coming
from unrelated code.
"""

from __future__ import annotations

import random
import warnings
from collections.abc import Sequence
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdMolDescriptors

from polyt5.chemistry import star_to_at

RDLogger.DisableLog("rdApp.*")

__all__ = [
    "ECFP_RADIUS",
    "FINGERPRINTS_AVAILABLE",
    "build_reference_fingerprints",
    "ecfp6",
    "loop_closed_mol",
    "max_similarity_to_reference",
    "tanimoto",
]

# ECFP6 is the diameter-6 variant, i.e. a Morgan fingerprint of radius 3.
ECFP_RADIUS = 3

_TERMINUS_SYMBOL = "At"


def _fingerprints_available() -> bool:
    """Probe once whether Morgan fingerprints can actually be computed here."""
    try:
        mol = Chem.MolFromSmiles("CCO")
        if mol is None:
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, ECFP_RADIUS, nBits=64)
        return True
    except Exception:
        return False


#: Whether ECFP fingerprints work in this process. A missing capability is
#: reported, never substituted with an approximation.
FINGERPRINTS_AVAILABLE: bool = _fingerprints_available()


def loop_closed_mol(psmiles: str) -> Chem.Mol | None:
    """Close a repeat unit into a loop, as the paper does before fingerprinting.

    The two ``[At]`` termini are deleted and a single bond is added between the
    atoms they were attached to, turning the repeat unit into a ring. The result
    approximates an infinite chain for fingerprinting purposes: every atom now
    sits in a cyclic environment, so no circular substructure is truncated by a
    chain end.

    Args:
        psmiles: A PSMILES repeat unit in either terminus notation.

    Returns:
        A sanitized ring-containing ``rdkit.Chem.Mol``, or ``None`` when the
        closure is not possible. This function never raises.

    Note:
        Three inputs legitimately return ``None`` rather than a molecule:

        * anything RDKit cannot parse, or that does not have exactly two
          monovalent astatine termini -- there is no pair of ends to join;
        * both termini attached to the *same* atom, which would ask for a bond
          from an atom to itself;
        * termini whose neighbours are *already* bonded, e.g. ``[At]CC[At]``
          (polyethylene). Closing that loop needs a two-membered ring, which
          RDKit cannot represent. Returning the un-closed molecule instead would
          silently mix two different fingerprint spaces in one comparison, so
          ``None`` is returned and the caller decides.
          # [AMBIGUITY] the paper does not say how it handles repeat units too
          # short to close; presumably none of its units were this short.
    """
    if not isinstance(psmiles, str) or not psmiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(star_to_at(psmiles))
        if mol is None:
            return None

        editable = Chem.RWMol(mol)
        terminus_indices = [
            atom.GetIdx()
            for atom in editable.GetAtoms()
            if atom.GetSymbol() == _TERMINUS_SYMBOL
        ]
        if len(terminus_indices) != 2:
            return None

        neighbours: list[int] = []
        for idx in terminus_indices:
            atom = editable.GetAtomWithIdx(idx)
            if atom.GetDegree() != 1 or atom.GetTotalValence() != 1:
                return None
            neighbours.append(atom.GetNeighbors()[0].GetIdx())

        if neighbours[0] == neighbours[1]:
            return None
        if editable.GetBondBetweenAtoms(neighbours[0], neighbours[1]) is not None:
            return None

        editable.AddBond(neighbours[0], neighbours[1], Chem.BondType.SINGLE)
        # Delete high index first so the low index stays valid.
        for idx in sorted(terminus_indices, reverse=True):
            editable.RemoveAtom(idx)

        closed = editable.GetMol()
        Chem.SanitizeMol(closed)
        return closed
    except Exception:
        return None


def ecfp6(psmiles: str, *, n_bits: int = 2048, loop_closed: bool = True):
    """Compute the paper's ECFP6 fingerprint for a polymer repeat unit.

    Args:
        psmiles: A PSMILES repeat unit in either terminus notation.
        n_bits: Fingerprint length. The paper uses 2048.
        loop_closed: Apply the paper's end-to-end loop closure first. Set to
            ``False`` only to fingerprint the repeat unit as written, which is
            *not* comparable with loop-closed fingerprints.

    Returns:
        An ``ExplicitBitVect``, or ``None`` if the structure cannot be parsed,
        the loop closure fails, or fingerprints are unavailable in this process.
        No fallback fingerprint is substituted.
    """
    if not FINGERPRINTS_AVAILABLE:
        return None
    if not isinstance(psmiles, str) or not psmiles.strip():
        return None

    if loop_closed:
        mol = loop_closed_mol(psmiles)
    else:
        try:
            mol = Chem.MolFromSmiles(star_to_at(psmiles))
        except Exception:
            mol = None
    if mol is None:
        return None

    try:
        with warnings.catch_warnings():
            # Local, not global: the modern generator API is unusable here (see
            # the module docstring), and its deprecation notice is expected.
            warnings.simplefilter("ignore")
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, ECFP_RADIUS, nBits=n_bits
            )
    except Exception:
        return None


def tanimoto(a: Any, b: Any) -> float:
    """Tanimoto similarity between two fingerprints.

    Args:
        a: A fingerprint from :func:`ecfp6`, or ``None``.
        b: A fingerprint from :func:`ecfp6`, or ``None``.

    Returns:
        The similarity in ``[0, 1]``. ``0.0`` is returned when either argument
        is ``None`` or the comparison fails, so a batch of comparisons never
        raises on one unfingerprintable member.

        # [AMBIGUITY] "no fingerprint" is reported as 0.0 rather than None
        # because the return type is a float used in aggregation; callers that
        # need to distinguish "dissimilar" from "unknown" should check for a
        # ``None`` fingerprint before calling.
    """
    if a is None or b is None:
        return 0.0
    try:
        return float(DataStructs.TanimotoSimilarity(a, b))
    except Exception:
        return 0.0


def build_reference_fingerprints(
    psmiles_list: Sequence[str], *, n_bits: int = 2048, loop_closed: bool = True
) -> list:
    """Fingerprint a reference set once, for repeated similarity queries.

    Args:
        psmiles_list: Known polymers, e.g. the training set.
        n_bits: Fingerprint length; must match the query side.
        loop_closed: Apply the paper's loop closure; must match the query side.

    Returns:
        The usable fingerprints. Entries that cannot be fingerprinted are
        dropped, so the result is usually shorter than the input and is *not*
        positionally aligned with it.
    """
    fingerprints = []
    for psmiles in psmiles_list:
        fingerprint = ecfp6(psmiles, n_bits=n_bits, loop_closed=loop_closed)
        if fingerprint is not None:
            fingerprints.append(fingerprint)
    return fingerprints


def max_similarity_to_reference(
    query_psmiles: str, reference_fps: Sequence[Any], *, n_bits: int = 2048
) -> float | None:
    """Similarity of a generated polymer to the closest known polymer.

    This is the paper's Figure 4B analysis: how close each generated candidate
    sits to its nearest neighbour in the known set, used to argue the model
    produces genuinely new chemistry rather than near-copies.

    Args:
        query_psmiles: The generated polymer.
        reference_fps: Fingerprints from :func:`build_reference_fingerprints`.
        n_bits: Fingerprint length; must match the reference side.

    Returns:
        The maximum Tanimoto similarity, or ``None`` when the answer is
        undefined -- an empty reference set, or a query that cannot be
        fingerprinted. ``None`` is not the same as ``0.0`` here: it means
        "not measurable", not "maximally dissimilar".
    """
    if not reference_fps:
        return None
    query_fp = ecfp6(query_psmiles, n_bits=n_bits)
    if query_fp is None:
        return None
    try:
        return float(max(DataStructs.TanimotoSimilarity(query_fp, ref) for ref in reference_fps))
    except Exception:
        return None


def mean_pairwise_tanimoto(
    psmiles_list: Sequence[str], *, max_pairwise: int = 200, seed: int = 0
) -> tuple[float | None, int]:
    """Mean Tanimoto over all pairs of a capped random sample.

    Pairwise similarity is O(n^2), so a large generated batch is subsampled
    before the comparison rather than computed in full.

    Args:
        psmiles_list: Polymers to compare, typically already deduplicated.
        max_pairwise: Cap on how many polymers enter the pairwise comparison.
        seed: Seed for the subsample, so a reported number is reproducible.

    Returns:
        A ``(mean, n_sampled)`` pair. The mean is ``None`` when fewer than two
        polymers could be fingerprinted; ``n_sampled`` is how many polymers
        entered the comparison.
    """
    unique = list(dict.fromkeys(psmiles_list))
    if len(unique) > max_pairwise:
        unique = random.Random(seed).sample(unique, max_pairwise)

    fingerprints = build_reference_fingerprints(unique)
    n_sampled = len(unique)
    if len(fingerprints) < 2:
        return None, n_sampled

    total = 0.0
    n_pairs = 0
    for i in range(len(fingerprints) - 1):
        try:
            sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[i + 1 :])
        except Exception:
            return None, n_sampled
        total += float(sum(sims))
        n_pairs += len(sims)
    if n_pairs == 0:
        return None, n_sampled
    return total / n_pairs, n_sampled
