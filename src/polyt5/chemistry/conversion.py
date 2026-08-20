"""Conversions between the four string types used by polyT5.

Four string types are kept strictly distinct throughout this package:

* **PSMILES** -- a SMILES string for a *polymer repeat unit* whose two chain
  ends (termini) are marked by placeholder atoms. polyT5 writes those
  placeholders as astatine, ``[At]``; other sources write them as ``[*]`` or a
  bare ``*``.
* **PSELFIES** -- the SELFIES string obtained by running ``selfies.encoder()``
  on an ``[At]``-capped PSMILES. The ``[At]`` substitution exists purely so the
  ``selfies`` package can encode the string at all: ``*`` is not part of the
  SELFIES alphabet, whereas astatine is a regular element.
* **SMILES** -- an ordinary small-molecule SMILES with no terminus semantics.
* **SELFIES** -- an ordinary small-molecule SELFIES.

``[At]`` is a *representation device* for chain ends and never a chemically
meaningful final product: no real polymer carries astatine end groups. Anything
that reports on real chemistry (synthetic accessibility, for example) must
replace the placeholders first.

Notation policy for this package: ``[At]`` is the internal canonical notation.
Every PSMILES-consuming function normalises ``[*]``/``*`` input to ``[At]``
first, so callers may pass either style.

Exception safety: every conversion in this module is total. Functions returning
``str | None`` return ``None`` on any failure instead of raising, and nothing
here ever prints. This layer is consumed by an RL reward function that is fed
raw model output, so it must never crash and must never pollute stdout.
"""

from __future__ import annotations

import re
from typing import Any

import selfies as sf
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import Mol

RDLogger.DisableLog("rdApp.*")

# Astatine stands in for the polymer chain ends; see the module docstring.
_ASTATINE_ATOMIC_NUMBER = 85

__all__ = [
    "CLEAVE_STRATEGIES",
    "STAR_TOKENS",
    "TERMINUS",
    "at_to_star",
    "cleave_and_cap",
    "count_termini",
    "cyclize_psmiles",
    "pselfies_to_psmiles",
    "psmiles_to_pselfies",
    "psmiles_to_pselfies_loop_break",
    "selfies_to_smiles",
    "smiles_to_selfies",
    "star_to_at",
]

#: Terminus spellings accepted on input, in the order they are substituted.
STAR_TOKENS: tuple[str, ...] = ("[*]", "*")

#: The terminus atom symbol polyT5 uses internally for polymer chain ends.
TERMINUS: str = "[At]"

# Matches "[*]" and the atom-mapped variants "[*:1]", "[1*]" some datasets emit.
_BRACKETED_STAR = re.compile(r"\[\d*\*(?::\d+)?\]")


def star_to_at(psmiles: str) -> str:
    """Rewrite ``*``-style polymer termini as ``[At]``.

    Handles ``[*]``, atom-mapped forms such as ``[*:1]``/``[1*]``, and the bare
    ``*`` token. Input that already uses ``[At]`` passes through unchanged, so
    the function is idempotent and safe to apply defensively.

    Args:
        psmiles: A PSMILES string in either terminus notation.

    Returns:
        The PSMILES with every terminus written as ``[At]``. Non-string input
        yields an empty string rather than raising, so the function is total
        against adversarial model output of any type.

    Examples:
        >>> star_to_at("[*]CCO[*]")
        '[At]CCO[At]'
        >>> star_to_at("*CCO*")
        '[At]CCO[At]'
    """
    if not isinstance(psmiles, str):
        return ""
    return _BRACKETED_STAR.sub(TERMINUS, psmiles).replace("*", TERMINUS)


def at_to_star(psmiles: str) -> str:
    """Rewrite ``[At]`` polymer termini as ``[*]``.

    Inverse of :func:`star_to_at` for the canonical ``[*]`` spelling; a bare
    ``*`` input is therefore not round-tripped back to a bare ``*``, and atom
    maps are not restored. Both are lossless as far as chemistry is concerned.

    Args:
        psmiles: A PSMILES string using ``[At]`` termini.

    Returns:
        The PSMILES with every ``[At]`` written as ``[*]``. Non-string input
        yields an empty string rather than raising.

    Examples:
        >>> at_to_star("[At]CCO[At]")
        '[*]CCO[*]'
    """
    if not isinstance(psmiles, str):
        return ""
    return psmiles.replace(TERMINUS, "[*]")


def psmiles_to_pselfies(psmiles: str) -> str | None:
    """Encode a polymer repeat unit as PSELFIES by DIRECT substitution.

    This implements the **direct** ``[*] -> [At]`` substitution: the termini are
    rewritten in place and the string is handed to ``selfies.encoder``. The
    repeat unit's own terminus positions are preserved exactly, which makes the
    transform invertible (:func:`pselfies_to_psmiles` recovers the input),
    deterministic, and cheap. It is this package's default for that reason.

    It is **not** the pipeline described in the paper. The paper joins the two
    ends into a ring, canonicalizes the ring, then cleaves a backbone bond and
    caps the cut with ``[At]``; because canonicalization erases where the
    termini originally sat, the cleaved bond may land elsewhere and yield a
    different -- equally correct -- repeat-unit phase of the same polymer. See
    :func:`psmiles_to_pselfies_loop_break` for that variant, and entry **A-01**
    of ``docs/ambiguity_register.md`` for the decision record.

    Star-style termini are normalised to ``[At]`` first, because ``*`` is not
    encodable by the ``selfies`` package.

    Args:
        psmiles: A PSMILES string in either terminus notation.

    Returns:
        The PSELFIES string, or ``None`` if the input cannot be encoded.

    Examples:
        >>> psmiles_to_pselfies("[At]CCO[At]")
        '[At][C][C][O][At]'
    """
    normalized = star_to_at(psmiles)
    if not normalized.strip():
        return None
    try:
        encoded = sf.encoder(normalized)
    except Exception:
        return None
    return encoded or None


def pselfies_to_psmiles(pselfies: str) -> str | None:
    """Decode PSELFIES back to a PSMILES repeat unit.

    Args:
        pselfies: A PSELFIES string.

    Returns:
        The decoded PSMILES with ``[At]`` termini, or ``None`` if the string
        cannot be decoded. Note that ``selfies.decoder`` silently returns an
        empty string for input containing no bracket tokens at all; that case is
        reported as ``None`` here rather than as an empty molecule.

    Examples:
        >>> pselfies_to_psmiles("[At][C][C][O][At]")
        '[At]CCO[At]'
    """
    if not isinstance(pselfies, str) or not pselfies.strip():
        return None
    try:
        decoded = sf.decoder(pselfies)
    except Exception:
        return None
    return decoded or None


def smiles_to_selfies(smiles: str) -> str | None:
    """Encode an ordinary molecule as SELFIES.

    No terminus logic is applied: ``*`` input is *not* rewritten to ``[At]``,
    and such input therefore fails to encode. Use :func:`psmiles_to_pselfies`
    for polymer repeat units.

    Args:
        smiles: An ordinary SMILES string.

    Returns:
        The SELFIES string, or ``None`` on failure.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        encoded = sf.encoder(smiles)
    except Exception:
        return None
    return encoded or None


def selfies_to_smiles(selfies_str: str) -> str | None:
    """Decode an ordinary SELFIES string back to SMILES.

    Args:
        selfies_str: An ordinary SELFIES string.

    Returns:
        The SMILES string, or ``None`` on failure.
    """
    if not isinstance(selfies_str, str) or not selfies_str.strip():
        return None
    try:
        decoded = sf.decoder(selfies_str)
    except Exception:
        return None
    return decoded or None


def count_termini(psmiles: str) -> int:
    """Count the polymer termini in a PSMILES string using RDKit.

    Counting is done on the parsed molecule rather than by string matching, so
    an ``[At]`` appearing inside an unparseable string is never miscounted.
    Star-style termini are normalised to ``[At]`` first, hence ``[*]CCO[*]``
    and ``[At]CCO[At]`` both count as two.

    Args:
        psmiles: A PSMILES string in either terminus notation.

    Returns:
        The number of terminus atoms, or ``0`` if the string is unparseable.

    Examples:
        >>> count_termini("[At]CCO[At]")
        2
        >>> count_termini("garbage")
        0
    """
    mol = _mol_from_psmiles(psmiles)
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "At")


def _mol_from_psmiles(psmiles: str):
    """Parse a PSMILES into an RDKit molecule, or ``None``.

    Internal helper shared by the rest of the package. Normalises star termini
    and swallows every RDKit failure mode, including the exceptions RDKit raises
    (rather than returning ``None``) for some malformed input.

    Args:
        psmiles: A PSMILES string in either terminus notation.

    Returns:
        A sanitized ``rdkit.Chem.Mol``, or ``None`` if parsing or sanitization
        failed.
    """
    normalized = star_to_at(psmiles)
    if not normalized.strip():
        return None
    try:
        return Chem.MolFromSmiles(normalized)
    except Exception:
        return None


# --------------------------------------------------------------------------
# The paper's loop-canonicalize-break conversion
# --------------------------------------------------------------------------
#
# Paper, verbatim: "the two polymer ends were first joined by removing the [*]
# tokens to form a cyclic structure, followed by canonicalization to mitigate
# any initial sequence bias. Subsequently, a single bond within the backbone was
# strategically cleaved, and Astatine (At) atoms were attached at the cleavage
# points."
#
# The three steps below implement that literally. They exist alongside -- not
# instead of -- the direct substitution in psmiles_to_pselfies; see A-01 in
# docs/ambiguity_register.md.

#: Bond-selection strategies accepted by :func:`cleave_and_cap`.
CLEAVE_STRATEGIES: tuple[str, ...] = ("canonical_rank", "original")

# Atom property marking the two atoms that the cyclization bond joined. Atom
# properties survive RWMol edits, so "original" can find its way back to the
# input's own terminus positions after the termini themselves are gone.
_JOIN_END_PROP = "_polyt5_join_end"


def cyclize_psmiles(psmiles: str) -> Mol | None:
    """Join a repeat unit's two ends into a ring, discarding the termini.

    Step one of the paper's pipeline: "the two polymer ends were first joined by
    removing the [*] tokens to form a cyclic structure". The two terminus atoms
    are deleted and a single bond is added between the heavy atoms they were
    attached to, so the repeat unit becomes the macrocycle whose ring bonds are
    exactly the polymer backbone.

    The two joined atoms are tagged with a private atom property so
    :func:`cleave_and_cap` can offer the ``"original"`` strategy. Canonicalization
    is *not* applied here as a string round trip -- that would discard the tags.
    It is applied in :func:`cleave_and_cap` through RDKit's canonical atom
    ranking, which is what makes the cleavage choice independent of how the
    input happened to be written.

    Args:
        psmiles: A PSMILES repeat unit in either terminus notation.

    Returns:
        A sanitized ring-bearing ``rdkit.Chem.Mol``, or ``None`` when the repeat
        unit cannot be cyclized. Never raises.

    Note:
        Cyclization is refused, and ``None`` returned, when:

        * the string does not carry exactly two termini, or a terminus is bonded
          to more than one atom (it is then not a chain end);
        * both termini hang off the **same** atom -- closing that loop needs a
          one-membered ring;
        * the two neighbouring atoms are **already bonded** -- closing that loop
          needs a two-membered ring.

        [AMBIGUITY] The paper does not mention these degenerate cases, and they
        are not exotic: every vinyl polymer is one. ``[At]CC[At]``
        (polyethylene), ``[At]CC(C)[At]`` (polypropylene) and ``[At]CC(Cl)[At]``
        (PVC) all have their two termini on adjacent, already-bonded carbons, so
        no molecular graph can express their cyclized form. The paper's corpus
        must have handled this somehow, but it does not say how;
        :func:`psmiles_to_pselfies_loop_break` falls back to the direct
        substitution for these inputs.
    """
    mol = _mol_from_psmiles(psmiles)
    if mol is None:
        return None

    terminus_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "At"]
    if len(terminus_indices) != 2:
        return None

    neighbours: list[int] = []
    for index in terminus_indices:
        bonded = [n.GetIdx() for n in mol.GetAtomWithIdx(index).GetNeighbors()]
        if len(bonded) != 1:  # a terminus must have valency one
            return None
        neighbours.append(bonded[0])

    first, second = neighbours
    if first == second:
        return None  # both ends on one atom: a one-membered ring
    if mol.GetBondBetweenAtoms(first, second) is not None:
        return None  # ends already bonded: a two-membered ring

    try:
        editable = Chem.RWMol(mol)
        editable.GetAtomWithIdx(first).SetBoolProp(_JOIN_END_PROP, True)
        editable.GetAtomWithIdx(second).SetBoolProp(_JOIN_END_PROP, True)
        editable.AddBond(first, second, Chem.BondType.SINGLE)
        # Remove high index first so the low index stays valid.
        for index in sorted(terminus_indices, reverse=True):
            editable.RemoveAtom(index)
        cyclized = editable.GetMol()
        Chem.SanitizeMol(cyclized)
    except Exception:
        return None
    return cyclized


def _cleavable_bonds(mol: Mol) -> list[Any]:
    """Return the bonds that may be cut to reopen a cyclized repeat unit.

    Candidates are single, non-aromatic, **ring** bonds. Ring membership is the
    load-bearing condition: after cyclization the entire backbone is the
    macrocycle, so cutting a ring bond reopens the chain and leaves one
    connected fragment with two loose ends. Cutting an *acyclic* bond would
    instead split the molecule into two disconnected pieces, which is not a
    repeat unit at all.

    Args:
        mol: The cyclized molecule.

    Returns:
        The candidate bonds, possibly empty.

    Note:
        [AMBIGUITY] Entry A-01 of ``docs/ambiguity_register.md`` currently
        describes the rule as the lowest canonical-rank backbone single bond
        "not in a ring". That is inverted relative to what the chemistry
        permits: after the ends are joined there are no acyclic backbone bonds
        left, and the only acyclic single bonds that remain belong to side
        chains, where a cut disconnects the molecule. The register entry needs
        correcting to "single, non-aromatic ring bond".
    """
    return [
        bond
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.SINGLE
        and not bond.GetIsAromatic()
        and bond.IsInRing()
    ]


def cleave_and_cap(mol: Mol, *, strategy: str = "canonical_rank") -> str | None:
    """Cut one backbone bond of a cyclized repeat unit and cap the ends.

    Steps two and three of the paper's pipeline: "a single bond within the
    backbone was strategically cleaved, and Astatine (At) atoms were attached at
    the cleavage points".

    Args:
        mol: A cyclized molecule, normally from :func:`cyclize_psmiles`.
        strategy: How to choose the bond.

            ``"canonical_rank"`` (default)
                Among the candidate bonds, take the one whose ``(min, max)``
                pair of RDKit canonical atom ranks is smallest. Canonical ranks
                depend only on the molecular graph, never on input atom order,
                so this is both deterministic and invariant across different
                SMILES writings of the same polymer -- which is precisely the
                "canonicalization to mitigate any initial sequence bias" the
                paper asks for. It is applied via
                ``Chem.CanonicalRankAtoms`` rather than a SMILES round trip so
                that the join tags survive.

            ``"original"``
                Cut the bond that :func:`cyclize_psmiles` created, returning the
                repeat unit to the input's own terminus positions. An identity
                check, not a generation strategy. Requires a molecule carrying
                the join tags; returns ``None`` for any other molecule.

    Returns:
        The canonical PSMILES of the reopened repeat unit, or ``None`` on
        failure. Never raises except for an invalid ``strategy``.

    Raises:
        ValueError: If ``strategy`` is not in :data:`CLEAVE_STRATEGIES`. This is
            a caller programming error, not adversarial data.

    Note:
        Returns ``None`` when no candidate bond exists -- an all-aromatic ring
        such as benzene, or an acyclic molecule. For any molecule produced by
        :func:`cyclize_psmiles` at least one candidate always exists, because
        the bond it adds is by construction a single, non-aromatic ring bond, so
        this branch is only reachable for hand-supplied molecules.
    """
    if strategy not in CLEAVE_STRATEGIES:
        raise ValueError(f"strategy must be one of {CLEAVE_STRATEGIES!r}, got {strategy!r}")
    if mol is None:
        return None

    try:
        candidates = _cleavable_bonds(mol)
    except Exception:
        return None
    if not candidates:
        return None

    try:
        if strategy == "original":
            chosen = _join_bond(mol)
            if chosen is None:
                return None
        else:
            ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))

            def rank_key(bond: Any) -> tuple[int, int]:
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                return (min(ranks[i], ranks[j]), max(ranks[i], ranks[j]))

            chosen = min(candidates, key=rank_key)

        begin, end = chosen.GetBeginAtomIdx(), chosen.GetEndAtomIdx()
        editable = Chem.RWMol(mol)
        editable.RemoveBond(begin, end)
        for index in (begin, end):
            terminus = editable.AddAtom(Chem.Atom(_ASTATINE_ATOMIC_NUMBER))
            editable.AddBond(index, terminus, Chem.BondType.SINGLE)
        reopened = editable.GetMol()
        Chem.SanitizeMol(reopened)
        return Chem.MolToSmiles(reopened) or None
    except Exception:
        return None


def _join_bond(mol: Mol) -> Any | None:
    """Return the bond created by :func:`cyclize_psmiles`, if it is still there.

    Args:
        mol: A molecule that may carry the join tags.

    Returns:
        The tagged bond, or ``None`` when the molecule was not produced by
        :func:`cyclize_psmiles`.
    """
    tagged = [a.GetIdx() for a in mol.GetAtoms() if a.HasProp(_JOIN_END_PROP)]
    if len(tagged) != 2:
        return None
    return mol.GetBondBetweenAtoms(tagged[0], tagged[1])


def psmiles_to_pselfies_loop_break(
    psmiles: str,
    *,
    strategy: str = "canonical_rank",
    fallback_to_direct: bool = True,
) -> str | None:
    """Encode a repeat unit as PSELFIES the way the paper describes.

    Runs the full pipeline: join the two ends into a ring, choose a backbone
    bond by a canonicalization-derived rule, cut it, cap both cut ends with
    ``[At]``, and encode the result.

    Because canonicalization erases where the input's termini sat, the bond that
    gets cut may differ from the original terminus position. The output is then
    a different *phase* of the same repeat unit -- the same infinite chain cut
    in a different place -- and so a different PSELFIES string for the same
    polymer. That is the intended behaviour, and the reason the paper
    canonicalizes at all: every writing of a polymer converges on one string.

    Args:
        psmiles: A PSMILES repeat unit in either terminus notation.
        strategy: Bond-selection strategy, see :func:`cleave_and_cap`.
        fallback_to_direct: What to do when the repeat unit cannot be cyclized
            (see :func:`cyclize_psmiles`). ``True`` (default) falls back to the
            direct substitution of :func:`psmiles_to_pselfies`, so a valid
            polymer always yields a valid PSELFIES. ``False`` returns ``None``
            instead, which is what you want when measuring how often the
            paper's pipeline actually applies.

    Returns:
        The PSELFIES string, or ``None`` on failure. Never raises except for an
        invalid ``strategy``.

    Raises:
        ValueError: If ``strategy`` is not in :data:`CLEAVE_STRATEGIES`.

    Examples:
        >>> psmiles_to_pselfies_loop_break("[At]CCO[At]")
        '[At][C][O][C][At]'
        >>> psmiles_to_pselfies("[At]CCO[At]")
        '[At][C][C][O][At]'
    """
    if strategy not in CLEAVE_STRATEGIES:
        raise ValueError(f"strategy must be one of {CLEAVE_STRATEGIES!r}, got {strategy!r}")

    def _fallback() -> str | None:
        return psmiles_to_pselfies(psmiles) if fallback_to_direct else None

    cyclized = cyclize_psmiles(psmiles)
    if cyclized is None:
        return _fallback()

    reopened = cleave_and_cap(cyclized, strategy=strategy)
    if reopened is None:
        return _fallback()

    return psmiles_to_pselfies(reopened)
