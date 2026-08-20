"""Aggregate chemistry metrics for a batch of generated polymers.

Reproduces the paper's evaluation cascade for hypothetical candidate
generation -- SMILES validity (SV), deduplication within the generated set
(DD), deduplication against the training set (TSD), polymer validity (PV), and
SELFIES reproducibility (SR) -- as plain Python data. Nothing here prints or
logs; results are returned as a frozen dataclass so the RLVR reward system can
consume them directly.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from rdkit import Chem, RDLogger

from .conversion import _mol_from_psmiles, star_to_at
from .novelty import NoveltyIndex
from .validity import selfies_reproducible, validate_pselfies

RDLogger.DisableLog("rdApp.*")

__all__ = [
    "SA_AVAILABLE",
    "GenerationChemMetrics",
    "evaluate_generations",
    "synthetic_accessibility",
]


def _load_sascorer() -> Any | None:
    """Import RDKit's Contrib SA_Score module, if it is available.

    The scorer ships with RDKit but lives outside the importable package tree,
    under ``RDConfig.RDContribDir/SA_Score``, so the directory has to be put on
    ``sys.path`` first. The import can still fail -- a source-only RDKit build,
    a trimmed wheel, or a platform policy blocking the fingerprint extension
    module the scorer depends on.

    Returns:
        The imported ``sascorer`` module, or ``None`` if it is unavailable.
    """
    try:
        from rdkit.Chem import RDConfig

        # Some Windows installations block RDKit's rdFingerprintGenerator
        # extension via Application Control policy, which makes sascorer
        # unimportable. Restore the original legacy code path when that
        # happens; see polyt5.chemistry._sa_compat for why this is faithful
        # rather than a substitute formula.
        from polyt5.chemistry._sa_compat import install_fingerprint_shim

        install_fingerprint_shim()

        contrib_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if contrib_path not in sys.path:
            sys.path.append(contrib_path)
        import sascorer  # type: ignore[import-not-found]

        return sascorer
    except Exception:
        return None


_SASCORER = _load_sascorer()

#: Whether RDKit's Contrib SA_Score module could be imported. When ``False``,
#: :func:`synthetic_accessibility` always returns ``None``. No substitute
#: formula is invented -- SA scores are only comparable against the published
#: RDKit implementation.
SA_AVAILABLE: bool = _SASCORER is not None


@dataclass(frozen=True)
class GenerationChemMetrics:
    """Chemistry statistics for one batch of generated PSELFIES strings.

    The counts nest the way the paper's filters do: the pool used for
    uniqueness and novelty is the *polymer-valid* pool, i.e. candidates that
    are both RDKit-valid and carry the expected number of termini (the paper's
    PV set).

    Attributes:
        n_generated: Number of strings in the batch.
        n_parseable: Strings that decoded to a non-empty PSMILES.
        n_valid: Strings whose decoded PSMILES RDKit parsed and sanitized
            (the paper's SV filter).
        n_correct_termini: Valid strings carrying exactly the expected number
            of ``[At]`` termini (the paper's PV filter).
        n_reproducible: Strings passing the strict SELFIES reproducibility
            test; see :func:`polyt5.chemistry.validity.selfies_reproducible`.
        n_unique: Distinct canonical PSMILES within the polymer-valid pool
            (the paper's DD filter).
        n_novel: Distinct polymer-valid candidates absent from the reference
            index (the paper's TSD filter). ``0`` when no index was supplied.
        validity_rate: ``n_valid / n_generated``.
        uniqueness_rate: ``n_unique / n_correct_termini``.
        novelty_rate: ``n_novel / n_unique``.
        reproducibility_rate: ``n_reproducible / n_generated``.
        duplicate_rate: ``1 - uniqueness_rate``.

    Note:
        Every rate is ``0.0`` when its denominator is zero, so an all-garbage
        batch scores zero everywhere rather than raising or producing NaN.
    """

    n_generated: int
    n_parseable: int
    n_valid: int
    n_correct_termini: int
    n_reproducible: int
    n_unique: int
    n_novel: int
    validity_rate: float
    uniqueness_rate: float
    novelty_rate: float
    reproducibility_rate: float
    duplicate_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Return the metrics as a JSON-serializable dict.

        Returns:
            A dict of plain ``int`` and ``float`` values, one key per field.
        """
        return asdict(self)


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Divide, treating a zero denominator as ``0.0``.

    Args:
        numerator: Dividend.
        denominator: Divisor; may be zero.

    Returns:
        The ratio, or ``0.0`` when ``denominator`` is zero.
    """
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def evaluate_generations(
    pselfies_list: Sequence[str],
    *,
    novelty_index: NoveltyIndex | None = None,
    expected_termini: int = 2,
) -> GenerationChemMetrics:
    """Score a batch of generated PSELFIES strings.

    Args:
        pselfies_list: Raw generated PSELFIES strings. Any element may be
            malformed; nothing in this function raises.
        novelty_index: Reference set of known polymers. When ``None``, novelty
            is not evaluated and both ``n_novel`` and ``novelty_rate`` are
            ``0.0`` -- absence of a reference is reported as zero novelty
            rather than as "everything is novel", so a missing index can never
            silently inflate an RL reward.
        expected_termini: Number of ``[At]`` termini a well-formed repeat unit
            must carry. Two for a linear polymer.

    Returns:
        A :class:`GenerationChemMetrics` describing the batch.
    """
    n_generated = len(pselfies_list)
    n_parseable = 0
    n_valid = 0
    n_correct_termini = 0
    n_reproducible = 0
    polymer_valid_canonical: list[str] = []

    for pselfies in pselfies_list:
        result = validate_pselfies(pselfies, expected_termini=expected_termini)
        if result.parseable:
            n_parseable += 1
        if result.valid:
            n_valid += 1
        if result.valid and result.correct_termini and result.canonical_psmiles is not None:
            n_correct_termini += 1
            polymer_valid_canonical.append(result.canonical_psmiles)
        # SR is a property of the raw generated string, so it is measured on
        # every candidate, valid or not.
        if selfies_reproducible(pselfies):
            n_reproducible += 1

    unique_canonical = set(polymer_valid_canonical)
    n_unique = len(unique_canonical)

    if novelty_index is None:
        n_novel = 0
    else:
        n_novel = sum(1 for psmiles in unique_canonical if novelty_index.is_novel(psmiles))

    uniqueness_rate = _safe_ratio(n_unique, n_correct_termini)
    return GenerationChemMetrics(
        n_generated=n_generated,
        n_parseable=n_parseable,
        n_valid=n_valid,
        n_correct_termini=n_correct_termini,
        n_reproducible=n_reproducible,
        n_unique=n_unique,
        n_novel=n_novel,
        validity_rate=_safe_ratio(n_valid, n_generated),
        uniqueness_rate=uniqueness_rate,
        novelty_rate=_safe_ratio(n_novel, n_unique),
        reproducibility_rate=_safe_ratio(n_reproducible, n_generated),
        duplicate_rate=0.0 if n_correct_termini <= 0 else 1.0 - uniqueness_rate,
    )


def _hydrogen_capped_mol(smiles_or_psmiles: str):
    """Parse a structure, replacing polymer termini with hydrogen.

    ``[At]`` is a representation device for chain ends, never a real end group,
    so scoring it as astatine would be chemically meaningless. Following the
    paper, each placeholder atom (``[At]``, or ``[*]``/``*`` dummy atoms) is
    turned into a hydrogen, which RDKit then folds into the neighbouring atom's
    implicit hydrogen count -- the repeat unit is scored as the corresponding
    monomer. Ordinary SMILES pass through untouched.

    Args:
        smiles_or_psmiles: A PSMILES or an ordinary SMILES string.

    Returns:
        A sanitized ``rdkit.Chem.Mol`` with hydrogen-capped ends, or ``None``
        if the input could not be parsed.
    """
    mol = _mol_from_psmiles(smiles_or_psmiles)
    if mol is None:
        return None
    try:
        editable = Chem.RWMol(mol)
        for atom in editable.GetAtoms():
            if atom.GetSymbol() == "At" or atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(1)
                atom.SetFormalCharge(0)
                atom.SetIsotope(0)
                atom.SetNoImplicit(False)
        capped = editable.GetMol()
        Chem.SanitizeMol(capped)
        return Chem.RemoveHs(capped)
    except Exception:
        return None


def synthetic_accessibility(smiles_or_psmiles: str) -> float | None:
    """Score how hard a structure is to synthesise (RDKit SA score, 1-10).

    Uses RDKit's Contrib ``sascorer`` implementation; no substitute formula is
    used when it is unavailable. Scores run from 1 (easy) to 10 (hard); the
    paper reports known polymers mostly in the 2-3 range and treats 6 as the
    threshold for structurally awkward candidates.

    For a PSMILES the ``[At]``/``*`` termini are replaced with hydrogen before
    scoring, so the repeat unit is scored as the corresponding monomer. This
    follows the paper, which computes SA "by first replacing the placeholder
    atoms ([At] or [*]) with hydrogen atoms, effectively treating each polymer
    as a monomer".

    Args:
        smiles_or_psmiles: A PSMILES (either terminus notation) or an ordinary
            SMILES string.

    Returns:
        The SA score, or ``None`` if the structure cannot be parsed or if
        :data:`SA_AVAILABLE` is ``False``.
    """
    if _SASCORER is None:
        return None
    mol = _hydrogen_capped_mol(star_to_at(smiles_or_psmiles))
    if mol is None:
        return None
    try:
        return float(_SASCORER.calculateScore(mol))
    except Exception:
        return None
