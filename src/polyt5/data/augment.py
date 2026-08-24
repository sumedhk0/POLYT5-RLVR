"""Split-safe invariance augmentation.

Training on N writings of one polymer with one target teaches the model that Tg
is a property of the molecule, not of the string, and multiplies the effective
training set at no labelling cost.

**The split boundary is the whole safety story.** :func:`augment_indices` takes
ONE split's index list and expands only those positions; every produced item
carries ``source_index`` back to the polymer it came from. A writing of a train
polymer appearing in test would be leakage indistinguishable from memorisation,
so ``tests/test_group_a_augment.py`` asserts the disjointness AND includes a
negative control that an augment-then-split ordering trips it.

The length filter is the corpus filter -- ``polyt5.data.prepare._count_tokens``
is imported rather than re-implemented, so an augmented writing can never be
admitted under a budget the original corpus would have rejected.

Torch-free.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from polyt5.chemistry.enumeration import enumerate_pselfies_writings
from polyt5.data.prepare import _count_tokens
from polyt5.data.tg_metadata import TgExample

__all__ = ["AugmentedExample", "augment_indices"]


@dataclass(frozen=True)
class AugmentedExample:
    """One writing of one polymer, with the corpus position it belongs to.

    Attributes:
        pselfies: The PSELFIES writing to train on.
        source_index: Position in the un-augmented example list. Every
            downstream target, descriptor vector and weight is looked up
            through this, so a writing can never acquire another polymer's
            label -- and a writing can never appear in a split its source does
            not belong to.
        is_original: Whether this is the corpus's own writing rather than an
            enumerated alternative.
    """

    pselfies: str
    source_index: int
    is_original: bool


def augment_indices(
    examples: Sequence[TgExample],
    indices: Iterable[int],
    *,
    n_writings: int,
    seed: int = 0,
    max_tokens: int = 200,
    tokenizer: object | None = None,
) -> list[AugmentedExample]:
    """Expand ONE split's indices into up to ``n_writings`` writings each.

    Args:
        examples: The full, un-augmented example list.
        indices: Positions belonging to the split being expanded -- typically
            one split's ``train`` list. Nothing outside this iterable is ever
            read, which is what keeps writings inside their own split.
        n_writings: Maximum writings per polymer, counting the original.
            ``1`` reproduces the un-augmented list exactly.
        seed: Reproducibility seed; each polymer uses ``seed + its index``.
        max_tokens: Token budget an alternative writing must also satisfy.
        tokenizer: Optional duck-typed tokenizer for the length count.

    Returns:
        The expanded list, grouped by source polymer in the order ``indices``
        gives, original writing first within each group.

    Raises:
        ValueError: If ``n_writings`` is below 1.
        IndexError: If an index is negative or out of range for ``examples``.
            Silently skipping it would quietly shrink a split, and silently
            wrapping a negative index would attach a writing to the wrong
            polymer's label.
    """
    if n_writings < 1:
        raise ValueError(f"n_writings must be >= 1, got {n_writings}")

    out: list[AugmentedExample] = []
    for position in indices:
        if position < 0:
            # A bare `examples[position]` would silently wrap a negative index
            # to the far end of the list instead of refusing it, quietly
            # attaching the wrong polymer's writings to `position`'s label.
            raise IndexError(f"augment_indices does not accept negative positions, got {position}")
        example = examples[position]  # IndexError on a bad index, deliberately
        out.append(
            AugmentedExample(pselfies=example.pselfies, source_index=position, is_original=True)
        )
        if n_writings == 1:
            continue

        seen = {example.pselfies}
        for pselfies in enumerate_pselfies_writings(
            example.row.psmiles, n_writings * 2, seed=seed + position
        ):
            if pselfies in seen or _count_tokens(pselfies, tokenizer) > max_tokens:
                continue
            seen.add(pselfies)
            out.append(
                AugmentedExample(
                    pselfies=pselfies, source_index=position, is_original=False
                )
            )
            if len(seen) >= n_writings:
                break
    return out
