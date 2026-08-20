"""The paper's four nested screening filters: SV, TSD, DD, PV.

Sahu et al. screen generated candidates through four filters applied in a fixed
order, each conditional on the previous one::

    SV   SMILES Validity            RDKit can parse and sanitize the structure
    TSD  Training Set Deduplication the candidate is not already in training
    DD   Dataset Deduplication      the candidate is not a repeat of an earlier
                                    candidate in this same generated batch
    PV   PSMILES Validity           exactly two astatine atoms, each of valency 1

    "These filters follow a nested relationship: SV > TSD > DD > PV."

Because the relationship is nested, every count is *conditional*: ``n_tsd``
counts candidates that passed SV **and** TSD, not candidates that would pass
TSD on their own. Each candidate is therefore charged to the *first* stage it
fails, and stages after that are never evaluated for it. This matters for
interpretation: a candidate that is both a training duplicate and structurally
malformed appears in the TSD bucket, and the PV bucket is not inflated by
candidates that were already rejected upstream.

PV is a **structural** rule, not a property rule. It is the one filter that
``polyt5.chemistry`` does not already cover: :func:`polyt5.chemistry.count_termini`
counts ``[At]`` atoms but does not check their valency, so a decoded string such
as ``[At]=C[At]`` -- two astatines, one of them divalent -- passes the count and
must still fail PV.

Everything in this module is total. Adversarial model output produces a
failure-tagged :class:`CandidateRecord`, never an exception and never a print.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from rdkit import Chem, RDLogger

from polyt5.chemistry import (
    SA_AVAILABLE,
    NoveltyIndex,
    pselfies_to_psmiles,
    selfies_reproducible,
    star_to_at,
    synthetic_accessibility,
    validate_psmiles,
)

# RDKit writes parse failures to stderr; this layer reports them structurally
# instead, and library code in this repository never prints.
RDLogger.DisableLog("rdApp.*")

__all__ = [
    "STAGES",
    "STAGE_DD",
    "STAGE_PV",
    "STAGE_SV",
    "STAGE_TSD",
    "CandidateRecord",
    "FilterCounts",
    "apply_filter_cascade",
    "has_valid_termini",
]

# Machine-readable stage tags, using the paper's own abbreviations so a metrics
# file can be read against the paper without a translation table.
STAGE_SV = "SV"
STAGE_TSD = "TSD"
STAGE_DD = "DD"
STAGE_PV = "PV"
STAGES: tuple[str, ...] = (STAGE_SV, STAGE_TSD, STAGE_DD, STAGE_PV)

# The paper's SA threshold for "structurally awkward": known polymers sit below
# 6, most of them in the 2-3 range.
SA_HARD_THRESHOLD = 6.0


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return ``numerator / denominator``, or ``0.0`` when the denominator is 0."""
    return 0.0 if denominator <= 0 else numerator / denominator


@dataclass(frozen=True)
class FilterCounts:
    """Candidate counts surviving each stage of the nested cascade.

    Attributes:
        n_input: Number of raw generated strings fed to the cascade.
        n_sv: Candidates whose structure RDKit accepted (SMILES Validity).
        n_tsd: Of those, the ones absent from the training set.
        n_dd: Of those, the first occurrence of each distinct polymer.
        n_pv: Of those, the ones with exactly the expected number of astatine
            termini, each of valency one (PSMILES Validity).

    Note:
        Every rate is expressed over ``n_input``, not over the previous stage,
        because the filters are nested: ``pv_rate`` is the fraction of *all*
        generated strings that survive the whole screen, which is the quantity
        the paper's yield figures report. Stage-conditional rates are recoverable
        from the raw counts by division.
    """

    n_input: int
    n_sv: int
    n_tsd: int
    n_dd: int
    n_pv: int

    @property
    def sv_rate(self) -> float:
        """Fraction of all inputs passing SV."""
        return _safe_ratio(self.n_sv, self.n_input)

    @property
    def tsd_rate(self) -> float:
        """Fraction of all inputs passing SV and TSD."""
        return _safe_ratio(self.n_tsd, self.n_input)

    @property
    def dd_rate(self) -> float:
        """Fraction of all inputs passing SV, TSD and DD."""
        return _safe_ratio(self.n_dd, self.n_input)

    @property
    def pv_rate(self) -> float:
        """Fraction of all inputs surviving the complete screen."""
        return _safe_ratio(self.n_pv, self.n_input)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view including the derived rates."""
        return {
            "n_input": self.n_input,
            "n_sv": self.n_sv,
            "n_tsd": self.n_tsd,
            "n_dd": self.n_dd,
            "n_pv": self.n_pv,
            "sv_rate": self.sv_rate,
            "tsd_rate": self.tsd_rate,
            "dd_rate": self.dd_rate,
            "pv_rate": self.pv_rate,
        }


@dataclass(frozen=True)
class CandidateRecord:
    """The full screening history of one generated candidate.

    Attributes:
        raw_pselfies: The model output exactly as generated.
        psmiles: The decoded PSMILES, or ``None`` if decoding failed.
        canonical_psmiles: Canonical PSMILES used as the deduplication key, or
            ``None`` if the structure was not RDKit-parseable.
        passed_sv: Passed SMILES Validity.
        passed_tsd: Passed SV *and* Training Set Deduplication.
        passed_dd: Passed SV, TSD *and* Dataset Deduplication.
        passed_pv: Passed the entire cascade.
        reproducible: The raw string survives the SELFIES round trip (the SR
            metric). Computed for every candidate, independently of the
            cascade, because the paper reports SR over all generated strings.
        failure_stage: The first stage the candidate failed -- one of ``"SV"``,
            ``"TSD"``, ``"DD"``, ``"PV"`` -- or ``None`` for a full pass. Stages
            after the failure are not evaluated, so their ``passed_*`` flags are
            ``False`` in the sense of "did not get through", not "was tested and
            rejected".
        sa_score: Synthetic accessibility score, or ``None`` when not requested,
            not available, or not computed for this candidate.
    """

    raw_pselfies: str
    psmiles: str | None
    canonical_psmiles: str | None
    passed_sv: bool
    passed_tsd: bool
    passed_dd: bool
    passed_pv: bool
    reproducible: bool
    failure_stage: str | None
    sa_score: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, suitable for ``generations.jsonl``."""
        return asdict(self)


def has_valid_termini(psmiles_or_mol: Any, *, expected: int = 2) -> bool:
    """Test the paper's PSMILES Validity (PV) rule.

    PV requires that a candidate "contained exactly two Astatine (At) atoms,
    each with a valency of one, as required by the polymer design rules". The
    valency clause is the part a terminus *count* misses: SELFIES decoding can
    emit a structure such as ``[At]=C[At]`` which has two astatines but one of
    them doubly bonded, so it cannot represent a chain end.

    An astatine terminus is accepted only when it is bonded to exactly one heavy
    atom by a single bond, carries no formal charge, and has no hydrogens --
    together, total valency one.

    Args:
        psmiles_or_mol: A PSMILES string in either terminus notation
            (``[At]`` or ``[*]``/``*``), or an already-parsed ``rdkit.Chem.Mol``.
        expected: Required number of termini. Two for a linear repeat unit.

    Returns:
        ``True`` if the structure has exactly ``expected`` monovalent astatine
        atoms. Any parse failure, wrong count, or bad valency returns ``False``;
        this function never raises.
    """
    mol = _as_mol(psmiles_or_mol)
    if mol is None:
        return False

    termini = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "At"]
    if len(termini) != expected:
        return False

    for atom in termini:
        try:
            if atom.GetFormalCharge() != 0:
                return False
            if atom.GetDegree() != 1:
                return False
            if atom.GetTotalNumHs() != 0:
                return False
            if atom.GetTotalValence() != 1:
                return False
        except Exception:
            return False
    return True


def _as_mol(psmiles_or_mol: Any):
    """Coerce a PSMILES string or Mol into a sanitized ``Mol``, or ``None``.

    Args:
        psmiles_or_mol: A PSMILES string, an ``rdkit.Chem.Mol``, or anything
            else (which yields ``None``).

    Returns:
        A sanitized ``rdkit.Chem.Mol``, or ``None`` if the input could not be
        interpreted as one.
    """
    if isinstance(psmiles_or_mol, Chem.Mol):
        mol = psmiles_or_mol
        try:
            # A Mol built with RWMol may not have its ring/valence caches filled.
            mol.UpdatePropertyCache(strict=False)
        except Exception:
            return None
        return mol

    if not isinstance(psmiles_or_mol, str) or not psmiles_or_mol.strip():
        return None
    try:
        return Chem.MolFromSmiles(star_to_at(psmiles_or_mol))
    except Exception:
        return None


def apply_filter_cascade(
    generated_pselfies: Sequence[str],
    *,
    training_index: NoveltyIndex | None,
    expected_termini: int = 2,
    compute_sa: bool = False,
) -> tuple[list[CandidateRecord], FilterCounts]:
    """Screen generated candidates through SV, TSD, DD and PV, in that order.

    The cascade is sequential and short-circuiting: a candidate that fails a
    stage is charged to that stage and no later stage is evaluated for it. The
    returned counts are therefore nested,
    ``n_input >= n_sv >= n_tsd >= n_dd >= n_pv``.

    Args:
        generated_pselfies: Raw model outputs, as PSELFIES strings. Order is
            preserved and is significant: the DD stage keeps the *first*
            writing of each distinct polymer and rejects later ones.
        training_index: Known training polymers for the TSD stage. When
            ``None`` the TSD stage is a no-op -- there is no reference set, so
            nothing can be a training duplicate -- and every SV survivor is
            passed straight to DD.
        expected_termini: Number of chain ends a well-formed repeat unit must
            have. Two for a linear polymer.
        compute_sa: Compute the synthetic accessibility score for candidates
            that survive the whole cascade. Off by default because SA scoring is
            the most expensive step here.

    Returns:
        A ``(records, counts)`` pair: one :class:`CandidateRecord` per input, in
        input order, and the aggregate :class:`FilterCounts`.

    Note:
        SELFIES that fail to decode are charged to SV. The paper defines SV as
        RDKit-based structural validity and does not say where a decoding
        failure lands, but a string that never becomes a molecule cannot be
        judged by any later filter, so SV is the only stage that can own it.
        # [AMBIGUITY] the paper does not state where undecodable SELFIES are counted.

        SA is computed only for full-pass candidates. The paper reports SA
        distributions for its *screened* candidate set, so scoring rejects would
        change the population being described.
        # [AMBIGUITY] the paper does not state the SA population explicitly.
    """
    records: list[CandidateRecord] = []
    seen_canonical: set[str] = set()
    n_sv = n_tsd = n_dd = n_pv = 0

    for raw in generated_pselfies:
        raw_str = raw if isinstance(raw, str) else str(raw)
        reproducible = _safe_reproducible(raw_str)

        psmiles = _safe_decode(raw_str)
        result = (
            validate_psmiles(psmiles, expected_termini=expected_termini)
            if psmiles is not None
            else None
        )

        # ------------------------------------------------------------ SV
        if result is None or not result.valid or result.canonical_psmiles is None:
            records.append(
                _record(raw_str, psmiles, None, reproducible, STAGE_SV, passed=0)
            )
            continue
        canonical = result.canonical_psmiles
        n_sv += 1

        # ----------------------------------------------------------- TSD
        if training_index is not None and canonical in training_index:
            records.append(
                _record(raw_str, psmiles, canonical, reproducible, STAGE_TSD, passed=1)
            )
            continue
        n_tsd += 1

        # ------------------------------------------------------------ DD
        if canonical in seen_canonical:
            records.append(
                _record(raw_str, psmiles, canonical, reproducible, STAGE_DD, passed=2)
            )
            continue
        seen_canonical.add(canonical)
        n_dd += 1

        # ------------------------------------------------------------ PV
        if not has_valid_termini(canonical, expected=expected_termini):
            records.append(
                _record(raw_str, psmiles, canonical, reproducible, STAGE_PV, passed=3)
            )
            continue
        n_pv += 1

        sa_score = None
        if compute_sa and SA_AVAILABLE:
            sa_score = _safe_sa(canonical)
        records.append(
            _record(
                raw_str, psmiles, canonical, reproducible, None, passed=4, sa_score=sa_score
            )
        )

    counts = FilterCounts(
        n_input=len(records), n_sv=n_sv, n_tsd=n_tsd, n_dd=n_dd, n_pv=n_pv
    )
    return records, counts


def _record(
    raw: str,
    psmiles: str | None,
    canonical: str | None,
    reproducible: bool,
    failure_stage: str | None,
    *,
    passed: int,
    sa_score: float | None = None,
) -> CandidateRecord:
    """Build a :class:`CandidateRecord` from the number of stages cleared.

    Args:
        raw: The raw generated string.
        psmiles: Decoded PSMILES, or ``None``.
        canonical: Canonical PSMILES, or ``None``.
        reproducible: SELFIES round-trip result for ``raw``.
        failure_stage: First failed stage, or ``None`` for a full pass.
        passed: How many of the four stages the candidate cleared (0-4).
        sa_score: Optional synthetic accessibility score.

    Returns:
        The assembled record.
    """
    return CandidateRecord(
        raw_pselfies=raw,
        psmiles=psmiles,
        canonical_psmiles=canonical,
        passed_sv=passed >= 1,
        passed_tsd=passed >= 2,
        passed_dd=passed >= 3,
        passed_pv=passed >= 4,
        reproducible=reproducible,
        failure_stage=failure_stage,
        sa_score=sa_score,
    )


def _safe_decode(raw: str) -> str | None:
    """Decode PSELFIES to PSMILES, mapping any internal failure to ``None``."""
    try:
        return pselfies_to_psmiles(raw)
    except Exception:
        return None


def _safe_reproducible(raw: str) -> bool:
    """Run the SELFIES round trip, mapping any internal failure to ``False``."""
    try:
        return bool(selfies_reproducible(raw))
    except Exception:
        return False


def _safe_sa(psmiles: str) -> float | None:
    """Score synthetic accessibility, mapping any internal failure to ``None``."""
    try:
        return synthetic_accessibility(psmiles)
    except Exception:
        return None
