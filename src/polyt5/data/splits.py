"""Deterministic, serializable dataset splits.

Paper splits (Sahu et al., npj AI 2026):

* Pretraining: 90% train / 10% held out. The paper does not sub-split the
  held-out 10%; we use 5% validation / 5% test (docs/data.md, register E-03).
* Property prediction: 80/20 train/test over FIVE random splits.
* Conditional generation: 90% train / 10% validation.

Every split is a seeded permutation partitioned by cumulative-rounded
boundaries, so each index appears exactly once and each part's size is within
one element of ``fraction * n``. Splits are saved as JSON so they can be
reproduced without rerunning the pipeline.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

__all__ = [
    "load_splits",
    "make_kfold_random_splits",
    "make_pretraining_splits",
    "random_split",
    "save_splits",
]

_FRACTION_TOLERANCE = 1e-6


def random_split(n: int, fractions: Sequence[float], *, seed: int) -> list[list[int]]:
    """Partition ``range(n)`` into deterministic random index lists.

    Args:
        n: Number of items to split.
        fractions: One fraction per part; must sum to 1 within tolerance.
        seed: Seed for the shuffling RNG; the same seed always yields the
            same split.

    Returns:
        One index list per fraction. Every index in ``range(n)`` appears in
        exactly one list, and each list's size is within one element of
        ``fraction * n`` (cumulative rounding).

    Raises:
        ValueError: ``fractions`` is empty, contains a negative value, or does
            not sum to 1.
    """
    if not fractions:
        raise ValueError("fractions must be non-empty")
    if any(f < 0 for f in fractions):
        raise ValueError(f"fractions must be non-negative, got {list(fractions)}")
    total = sum(fractions)
    if abs(total - 1.0) > _FRACTION_TOLERANCE:
        raise ValueError(f"fractions must sum to 1, got {list(fractions)} (sum={total})")

    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    parts: list[list[int]] = []
    cumulative = 0.0
    start = 0
    for fraction in fractions:
        cumulative += fraction
        end = round(cumulative * n)
        parts.append(indices[start:end])
        start = end
    parts[-1].extend(indices[start:])  # guard against float rounding at the tail
    return parts


def make_pretraining_splits(
    n: int,
    *,
    seed: int,
    train: float = 0.9,
    val: float = 0.05,
    test: float = 0.05,
) -> dict[str, list[int]]:
    """Build the pretraining split: paper 90/10, our 10% split into 5/5.

    Args:
        n: Corpus size.
        seed: RNG seed.
        train: Train fraction (paper: 0.9).
        val: Validation fraction (ours; the paper leaves the 10% unsplit).
        test: Test fraction (ours).

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}``.
    """
    train_idx, val_idx, test_idx = random_split(n, [train, val, test], seed=seed)
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def make_kfold_random_splits(
    n: int,
    *,
    k: int = 5,
    train_fraction: float = 0.8,
    base_seed: int = 0,
) -> list[tuple[list[int], list[int]]]:
    """Build the paper's "five random splits" for property prediction.

    These are k independent random 80/20 splits (seeds ``base_seed`` through
    ``base_seed + k - 1``), not a partitioning k-fold cross-validation --
    matching the paper's wording "five random splits".

    Args:
        n: Dataset size.
        k: Number of random splits (paper: 5).
        train_fraction: Train fraction per split (paper: 0.8).
        base_seed: Seed of the first split; split ``i`` uses ``base_seed + i``.

    Returns:
        ``[(train_indices, test_indices), ...]`` of length ``k``.
    """
    folds: list[tuple[list[int], list[int]]] = []
    for i in range(k):
        train_idx, test_idx = random_split(
            n, [train_fraction, 1.0 - train_fraction], seed=base_seed + i
        )
        folds.append((train_idx, test_idx))
    return folds


def save_splits(path: str | Path, splits: dict) -> Path:
    """Serialize a splits dict to JSON.

    Args:
        path: Destination file path (parent directories are created).
        splits: JSON-serializable mapping, e.g. name -> index list, or any
            nested structure of lists/dicts/numbers.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    return path


def load_splits(path: str | Path) -> dict:
    """Load a splits dict previously written by :func:`save_splits`.

    Note:
        JSON has no tuple type, so tuples saved via :func:`save_splits` come
        back as lists.

    Args:
        path: Path to the JSON file.

    Returns:
        The deserialized dict.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))
