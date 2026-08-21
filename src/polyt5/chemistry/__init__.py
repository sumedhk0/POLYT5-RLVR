"""Chemistry layer for polyT5: polymer string conversion, validity, and metrics.

This package is the shared chemical foundation for both the polyT5
reproduction and the RLVR extension. It has **zero dependency on torch or on
any model code** -- only ``rdkit`` and ``selfies`` -- so the RL reward system
can import it cheaply, and every entry point is total: adversarial model output
produces ``None``, ``False``, or a failure-tagged result object, never an
exception and never a print.

Four string types are kept strictly distinct:

===========  ============================================================
Type         Meaning
===========  ============================================================
PSMILES      SMILES for a polymer repeat unit, chain ends marked ``[At]``
             (``[*]``/``*`` on input is normalised to ``[At]``).
PSELFIES     ``selfies.encoder()`` applied to an ``[At]``-capped PSMILES.
SMILES       An ordinary small-molecule SMILES, no terminus semantics.
SELFIES      An ordinary small-molecule SELFIES.
===========  ============================================================

``[At]`` marks where the repeat unit joins its neighbours. It is a
representation device only -- chosen because ``selfies`` cannot encode the
``*`` wildcard but handles astatine as a regular element -- and never a
chemically meaningful final product.

Example:
    >>> from polyt5.chemistry import psmiles_to_pselfies, validate_pselfies
    >>> psmiles_to_pselfies("[*]CCO[*]")
    '[At][C][C][O][At]'
    >>> validate_pselfies("[At][C][C][O][At]").valid
    True
"""

from __future__ import annotations

from .canonicalization import canonical_pselfies, canonical_psmiles, is_same_polymer
from .conversion import (
    CLEAVE_STRATEGIES,
    STAR_TOKENS,
    TERMINUS,
    at_to_star,
    cleave_and_cap,
    count_termini,
    cyclize_psmiles,
    pselfies_to_psmiles,
    psmiles_to_pselfies,
    psmiles_to_pselfies_loop_break,
    selfies_to_smiles,
    smiles_to_selfies,
    star_to_at,
)
from .metrics import (
    SA_AVAILABLE,
    GenerationChemMetrics,
    evaluate_generations,
    synthetic_accessibility,
)
from .novelty import NoveltyIndex, novelty_rate
from .scalable_novelty import (
    HASH_NAME,
    HASH_SPACES,
    HashSpaceMismatchError,
    NoveltyIndexMetadata,
    ScalableNoveltyIndex,
    hash64,
    hash64_many,
    index_paths,
)
from .validity import ValidityResult, selfies_reproducible, validate_pselfies, validate_psmiles

__all__ = [
    "CLEAVE_STRATEGIES",
    "HASH_NAME",
    "HASH_SPACES",
    "SA_AVAILABLE",
    "STAR_TOKENS",
    "TERMINUS",
    "GenerationChemMetrics",
    "HashSpaceMismatchError",
    "NoveltyIndex",
    "NoveltyIndexMetadata",
    "ScalableNoveltyIndex",
    "ValidityResult",
    "at_to_star",
    "canonical_pselfies",
    "canonical_psmiles",
    "cleave_and_cap",
    "count_termini",
    "cyclize_psmiles",
    "evaluate_generations",
    "hash64",
    "hash64_many",
    "index_paths",
    "is_same_polymer",
    "novelty_rate",
    "pselfies_to_psmiles",
    "psmiles_to_pselfies",
    "psmiles_to_pselfies_loop_break",
    "selfies_reproducible",
    "selfies_to_smiles",
    "smiles_to_selfies",
    "star_to_at",
    "synthetic_accessibility",
    "validate_pselfies",
    "validate_psmiles",
]
