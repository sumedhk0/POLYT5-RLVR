"""Novelty of generated polymers against a reference set.

Novelty is the paper's TSD filter (training-set deduplication): a generated
repeat unit is novel when it does not already appear in the training data.
Membership is decided on canonical PSMILES, so a candidate written from the
other chain end, or with ``[*]`` instead of ``[At]`` termini, is correctly
recognised as already known.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .canonicalization import canonical_psmiles
from .conversion import pselfies_to_psmiles

__all__ = ["NoveltyIndex", "novelty_rate"]


class NoveltyIndex:
    """A set of known polymers, queried by canonical PSMILES.

    Reference entries that cannot be canonicalized are skipped rather than
    stored, so a malformed training row never becomes an unmatchable ghost
    entry. The index is therefore usually smaller than the input iterable, and
    duplicate writings of one polymer collapse to a single entry.
    """

    def __init__(self, reference_psmiles: Iterable[str]) -> None:
        """Build an index from reference PSMILES strings.

        Args:
            reference_psmiles: Known polymers, in either terminus notation.
                Unparseable entries are silently skipped.
        """
        self._canonical: set[str] = set()
        for entry in reference_psmiles:
            canonical = canonical_psmiles(entry)
            if canonical is not None:
                self._canonical.add(canonical)

    @classmethod
    def from_pselfies(cls, reference_pselfies: Iterable[str]) -> NoveltyIndex:
        """Build an index from reference PSELFIES strings.

        Args:
            reference_pselfies: Known polymers as PSELFIES. Entries that fail to
                decode are silently skipped.

        Returns:
            A :class:`NoveltyIndex` over the decoded polymers.
        """
        decoded = (pselfies_to_psmiles(entry) for entry in reference_pselfies)
        return cls([psmiles for psmiles in decoded if psmiles is not None])

    def __contains__(self, psmiles: str) -> bool:
        """Test whether a polymer is already known.

        Args:
            psmiles: Candidate PSMILES, in either terminus notation.

        Returns:
            ``True`` if the candidate's canonical form is in the index. A
            candidate that cannot be canonicalized is never in the index and so
            returns ``False``.
        """
        canonical = canonical_psmiles(psmiles)
        return canonical is not None and canonical in self._canonical

    def __len__(self) -> int:
        """Return the number of distinct known polymers in the index."""
        return len(self._canonical)

    def is_novel(self, psmiles: str) -> bool:
        """Test whether a candidate is absent from the reference set.

        Args:
            psmiles: Candidate PSMILES, in either terminus notation.

        Returns:
            ``True`` if the candidate is not in the index.

        Note:
            An unparseable candidate cannot match any index entry and is
            therefore reported as novel. Novelty is meaningless for invalid
            structures, so callers must gate on validity first;
            :func:`polyt5.chemistry.metrics.evaluate_generations` does exactly
            that.
        """
        return psmiles not in self

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={len(self)})"


def novelty_rate(candidates: Sequence[str], index: NoveltyIndex) -> float:
    """Fraction of candidates absent from the reference set.

    Args:
        candidates: Candidate PSMILES strings. Duplicates within the sequence
            are *not* collapsed; pass a deduplicated sequence if you want
            novelty per distinct polymer.
        index: The reference set to check against.

    Returns:
        Novel candidates divided by total candidates, or ``0.0`` for an empty
        sequence.
    """
    if not candidates:
        return 0.0
    n_novel = sum(1 for candidate in candidates if index.is_novel(candidate))
    return n_novel / len(candidates)
