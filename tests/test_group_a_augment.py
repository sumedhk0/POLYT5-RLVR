"""Group A Task 4: invariance augmentation, and the split boundary it must not cross."""

from __future__ import annotations

import pytest

from polyt5.chemistry import canonical_psmiles, pselfies_to_psmiles
from polyt5.chemistry.enumeration import enumerate_pselfies_writings
from polyt5.data.augment import AugmentedExample, augment_indices
from polyt5.data.tg_metadata import TgExample, TgRow

PSMILES = [
    "[*]CC(C)(C(=O)OC)[*]",
    "[*]CCO[*]",
    "[*]CC(c1ccccc1)[*]",
    "[*]CC(Cl)[*]",
    "[*]CC(C#N)[*]",
    "[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]",
]


def build_examples() -> list[TgExample]:
    from polyt5.chemistry import psmiles_to_pselfies, star_to_at

    out: list[TgExample] = []
    for index, psmiles in enumerate(PSMILES):
        pselfies = psmiles_to_pselfies(star_to_at(psmiles))
        assert pselfies is not None, psmiles
        out.append(
            TgExample(
                pselfies=pselfies,
                row=TgRow(
                    psmiles=psmiles, tg=300.0 + index, std=0.0, num_of_points=1,
                    reliability="black", polymer_class="test", descriptors=(float(index),),
                ),
            )
        )
    return out


def canonical_of(pselfies: str) -> str | None:
    psmiles = pselfies_to_psmiles(pselfies)
    return None if psmiles is None else canonical_psmiles(psmiles)


# ------------------------------------------------------------------- enumeration
def test_enumeration_produces_several_distinct_writings_of_one_polymer():
    writings = enumerate_pselfies_writings("[*]CC(C)(C(=O)OC)[*]", 5, seed=3)
    assert len(writings) >= 2
    assert len(set(writings)) == len(writings)


def test_every_writing_canonicalises_back_to_the_same_polymer():
    reference = canonical_psmiles("[*]CC(C)(C(=O)OC)[*]")
    for writing in enumerate_pselfies_writings("[*]CC(C)(C(=O)OC)[*]", 6, seed=3):
        assert canonical_of(writing) == reference


def test_enumeration_is_deterministic_including_at_seed_zero():
    """RDKit's randomSeed=0 means 'pick a random seed'; we must not pass it through."""
    first = enumerate_pselfies_writings("[*]CC(c1ccccc1)[*]", 6, seed=0)
    second = enumerate_pselfies_writings("[*]CC(c1ccccc1)[*]", 6, seed=0)
    assert first == second
    assert first != enumerate_pselfies_writings("[*]CC(c1ccccc1)[*]", 6, seed=1) or len(first) <= 1


def test_enumeration_returns_empty_on_junk_instead_of_raising():
    assert enumerate_pselfies_writings("not a molecule at all", 4) == []
    assert enumerate_pselfies_writings("", 4) == []


def test_enumeration_refuses_a_nonpositive_count():
    with pytest.raises(ValueError, match="n_writings"):
        enumerate_pselfies_writings("[*]CCO[*]", 0)


# ------------------------------------------------------------------ augmentation
def test_n_writings_one_reproduces_the_unaugmented_corpus_exactly():
    examples = build_examples()
    out = augment_indices(examples, [0, 2, 4], n_writings=1)
    assert [a.pselfies for a in out] == [examples[i].pselfies for i in (0, 2, 4)]
    assert all(a.is_original for a in out)
    assert [a.source_index for a in out] == [0, 2, 4]


def test_augmentation_multiplies_the_train_set():
    examples = build_examples()
    out = augment_indices(examples, [0, 1, 2], n_writings=4, seed=5)
    assert len(out) > 3
    assert all(isinstance(a, AugmentedExample) for a in out)
    assert sum(1 for a in out if a.is_original) == 3


def test_every_writing_points_back_at_a_requested_index():
    examples = build_examples()
    train = [0, 1, 2, 3]
    out = augment_indices(examples, train, n_writings=4, seed=5)
    assert {a.source_index for a in out} <= set(train)


def test_no_augmented_train_writing_lands_in_the_test_split():
    """The leakage check. The corpus is deduplicated on canonical PSMILES, so a
    train writing sharing a test polymer's canonical form can only mean the
    augmentation crossed the boundary."""
    examples = build_examples()
    train, test = [0, 1, 2], [3, 4, 5]
    train_canonical = {canonical_of(a.pselfies)
                       for a in augment_indices(examples, train, n_writings=6, seed=5)}
    test_canonical = {canonical_of(examples[i].pselfies) for i in test}
    assert train_canonical.isdisjoint(test_canonical)


def test_the_leakage_check_catches_augment_before_split():
    """Negative control ON THE GUARD, not on ``augment_indices`` itself.

    ``test_no_augmented_train_writing_lands_in_the_test_split`` above is the
    real safety net: it always calls ``augment_indices`` with ONLY the train
    indices, so it can only ever fail if ``augment_indices`` itself leaks
    across the boundary it was given. This test exists to prove that same
    disjointness assertion is capable of firing at all -- that it is not
    vacuous -- by reproducing spec 4.3's ordering mistake explicitly:
    augment the WHOLE corpus in one call (mixing train and test), THEN split
    the flat output afterwards by position, instead of splitting the original
    indices first and augmenting each split separately.

    Train and test are interleaved here ([0, 2, 4] / [1, 3, 5]), matching how
    a real split (``random_split``) actually looks -- train is not the first
    contiguous block. A caller who forgot to split first and instead slices
    the augmented list by the ORIGINAL train/test proportion (the natural
    mistake: "train was half the corpus before, so take the first half now")
    lands on a slice that includes an entire test polymer's group, because
    that polymer's writings sit inside the "first half" of the interleaved
    order. This must trip the same disjointness assertion the real guard
    uses; if it does not, the assertion has no teeth.
    """
    examples = build_examples()
    train, test = [0, 2, 4], [1, 3, 5]

    # The mistake: augment every index in one call, not just train's.
    everything = augment_indices(examples, range(len(examples)), n_writings=6, seed=5)

    # The mistake compounded: split the flat output by POSITION afterwards,
    # using the original train/test proportion, instead of re-deriving the
    # split from each item's `source_index`.
    naive_train_fraction = len(train) / len(examples)
    wrong_train = everything[: round(len(everything) * naive_train_fraction)]

    test_canonical = {canonical_of(examples[i].pselfies) for i in test}
    wrong_canonical = {canonical_of(a.pselfies) for a in wrong_train}
    assert not wrong_canonical.isdisjoint(test_canonical), (
        "the leakage assertion must FAIL on an augment-then-split ordering, or it is "
        "not testing anything"
    )


def test_a_writing_longer_than_the_token_budget_is_dropped():
    examples = build_examples()
    out = augment_indices(examples, [0], n_writings=6, seed=5, max_tokens=4)
    assert all(a.is_original for a in out), "only the original survives a 4-token budget"


def test_out_of_range_index_is_refused():
    examples = build_examples()
    with pytest.raises(IndexError):
        augment_indices(examples, [99], n_writings=2)


def test_negative_index_is_refused_not_silently_wrapped():
    """A bare ``examples[-1]`` would silently return the LAST example instead
    of raising, attaching that polymer's writings to the wrong ``source_index``."""
    examples = build_examples()
    with pytest.raises(IndexError):
        augment_indices(examples, [-1], n_writings=2)
