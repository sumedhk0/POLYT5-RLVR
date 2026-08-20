"""Validity checks for generated polymer strings.

Mirrors the filters the polyT5 paper applies to generated candidates:

* **SV (SMILES validity)** -- RDKit parses and sanitizes the structure.
* **PV (PSMILES validity)** -- the structure carries exactly two ``[At]``
  termini, i.e. it really is a repeat unit and not an arbitrary molecule.
* **SR (SELFIES reproducibility)** -- the generated SELFIES string survives a
  conversion to SMILES and back unchanged; see :func:`selfies_reproducible`.

Every entry point is total: adversarial model output produces a
:class:`ValidityResult` with ``valid=False`` and a machine-readable ``reason``,
never an exception.
"""

from __future__ import annotations

from dataclasses import dataclass

import selfies as sf
from rdkit import RDLogger

from .canonicalization import canonical_psmiles
from .conversion import count_termini, pselfies_to_psmiles, psmiles_to_pselfies

RDLogger.DisableLog("rdApp.*")

__all__ = ["ValidityResult", "selfies_reproducible", "validate_pselfies", "validate_psmiles"]

# Machine-readable failure tags. Kept as literals so downstream reward shaping
# can branch on them without string surprises.
REASON_EMPTY_INPUT = "empty_input"
REASON_DECODE_FAILED = "decode_failed"
REASON_RDKIT_PARSE_FAILED = "rdkit_parse_failed"
REASON_WRONG_TERMINI_COUNT = "wrong_termini_count"


@dataclass(frozen=True)
class ValidityResult:
    """Outcome of validating one generated polymer string.

    Attributes:
        valid: RDKit parsed the PSMILES and sanitization succeeded. This is the
            paper's SV criterion and is independent of the terminus count.
        parseable: The string got as far as producing a non-empty PSMILES. For
            PSELFIES input this means ``selfies.decoder`` returned something;
            for PSMILES input, which has no decode stage, it means the input
            string was non-empty.
        n_termini: Number of ``[At]`` terminus atoms found.
        correct_termini: ``n_termini == expected_termini``. Combined with
            ``valid`` this is the paper's PV criterion.
        canonical_psmiles: Canonical PSMILES, or ``None`` when invalid. This is
            the key to use for deduplication.
        reason: Short machine-readable failure tag, or ``None`` when the string
            is both valid and correctly terminated. One of ``"empty_input"``,
            ``"decode_failed"``, ``"rdkit_parse_failed"``,
            ``"wrong_termini_count"``.
    """

    valid: bool
    parseable: bool
    n_termini: int
    correct_termini: bool
    canonical_psmiles: str | None
    reason: str | None


def validate_pselfies(pselfies: str, *, expected_termini: int = 2) -> ValidityResult:
    """Validate a generated PSELFIES string.

    Decodes to PSMILES, then applies the same checks as
    :func:`validate_psmiles`.

    Args:
        pselfies: The PSELFIES string to check, typically raw model output.
        expected_termini: How many ``[At]`` termini a well-formed repeat unit
            must have. Two for a linear polymer.

    Returns:
        A :class:`ValidityResult`. A string that cannot be decoded at all gets
        ``reason="decode_failed"``.
    """
    psmiles = pselfies_to_psmiles(pselfies)
    if psmiles is None:
        return ValidityResult(
            valid=False,
            parseable=False,
            n_termini=0,
            correct_termini=False,
            canonical_psmiles=None,
            reason=REASON_DECODE_FAILED,
        )
    return _validate_decoded(psmiles, expected_termini=expected_termini)


def validate_psmiles(psmiles: str, *, expected_termini: int = 2) -> ValidityResult:
    """Validate a PSMILES repeat unit.

    Star-style termini are normalised to ``[At]`` before checking, so both
    notations are accepted.

    Args:
        psmiles: The PSMILES string to check.
        expected_termini: How many ``[At]`` termini a well-formed repeat unit
            must have. Two for a linear polymer.

    Returns:
        A :class:`ValidityResult`. An empty or whitespace-only string gets
        ``reason="empty_input"``; a string RDKit rejects gets
        ``reason="rdkit_parse_failed"``.
    """
    if not isinstance(psmiles, str) or not psmiles.strip():
        return ValidityResult(
            valid=False,
            parseable=False,
            n_termini=0,
            correct_termini=False,
            canonical_psmiles=None,
            reason=REASON_EMPTY_INPUT,
        )
    return _validate_decoded(psmiles, expected_termini=expected_termini)


def _validate_decoded(psmiles: str, *, expected_termini: int) -> ValidityResult:
    """Run the RDKit and terminus checks on a non-empty PSMILES string.

    Args:
        psmiles: A non-empty PSMILES string.
        expected_termini: Required number of ``[At]`` termini.

    Returns:
        A :class:`ValidityResult` with ``parseable=True``.

    Note:
        The paper's PV filter additionally requires each ``[At]`` to have a
        valency of one. That extra check is not folded into ``correct_termini``
        here; see the module TODO in ``metrics.py``. In practice a decoded
        SELFIES string cannot give astatine a higher valency, since SELFIES
        enforces valence constraints during decoding.
    """
    canonical = canonical_psmiles(psmiles)
    if canonical is None:
        return ValidityResult(
            valid=False,
            parseable=True,
            n_termini=0,
            correct_termini=False,
            canonical_psmiles=None,
            reason=REASON_RDKIT_PARSE_FAILED,
        )

    n_termini = count_termini(psmiles)
    correct_termini = n_termini == expected_termini
    return ValidityResult(
        valid=True,
        parseable=True,
        n_termini=n_termini,
        correct_termini=correct_termini,
        canonical_psmiles=canonical,
        reason=None if correct_termini else REASON_WRONG_TERMINI_COUNT,
    )


def selfies_reproducible(pselfies: str, *, strict: bool = True) -> bool:
    """Test whether a generated PSELFIES string is reproducible (the SR metric).

    Args:
        pselfies: The PSELFIES string to test, typically raw model output.
        strict: Select the round trip to apply. ``True`` (default) is the
            paper's string-identity definition; ``False`` is the weaker
            semantic definition. See the NOTE below.

    Returns:
        ``True`` if the string passes the selected round trip. Any failure
        along the way -- undecodable string, unparseable molecule -- returns
        ``False`` rather than raising.

    Note:
        **Which round trip is implemented, and where the ambiguity lies.**

        The paper defines SR as "the fraction of generated SELFIES strings
        that, when converted to SMILES and back to SELFIES, yield the identical
        string" (Sahu et al. 2026, ablation section). Taken literally that is
        string identity::

            selfies.encoder(selfies.decoder(s)) == s

        which is what ``strict=True`` implements and what this package uses as
        the default, because it is the paper's own wording and it is the metric
        that discriminates -- roughly 19% of the paper's screened candidates
        pass it (3,978 of 21,457).

        The wording is nonetheless ambiguous on one point: whether the
        intermediate SMILES is canonicalized before re-encoding. It is not
        canonicalized here. Canonicalizing would test something strictly
        weaker, namely whether the *molecule* survives the round trip rather
        than whether the *string* does, and essentially every RDKit-valid
        string passes that. ``strict=False`` implements that weaker reading --
        re-encode the decoded PSMILES and check the result still decodes to the
        same canonical PSMILES -- and is exposed only so callers can measure
        the gap between the two definitions.

        Concretely, ``[At][C][Branch1][Ring2][C][C][At]`` decodes to
        ``[At]CCC[At]`` (the branch-length token overruns the remaining symbols,
        so the branch collapses into the main chain) and re-encodes to
        ``[At][C][C][C][At]``. It fails ``strict=True`` and passes
        ``strict=False``.

        SR is a property of the *string*, so it is measured on the raw
        generated PSELFIES, never on a canonicalized form.
    """
    if not isinstance(pselfies, str) or not pselfies.strip():
        return False

    psmiles = pselfies_to_psmiles(pselfies)
    if psmiles is None:
        return False

    if strict:
        try:
            return sf.encoder(psmiles) == pselfies
        except Exception:
            return False

    reencoded = psmiles_to_pselfies(psmiles)
    if reencoded is None:
        return False
    redecoded = pselfies_to_psmiles(reencoded)
    if redecoded is None:
        return False
    original = canonical_psmiles(psmiles)
    return original is not None and original == canonical_psmiles(redecoded)
