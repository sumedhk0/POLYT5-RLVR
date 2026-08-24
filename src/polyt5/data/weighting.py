"""Per-example training weights from the curated measurement provenance.

Every LamaLab row carries ``meta.num_of_points``, ``meta.std`` and
``meta.reliability``, all of which the baseline fine-tune ignores. 279 polymers
have repeated measurements; their spread has median 5.6 K and reaches 145 K.
A 145 K label is close to noise, and training on it at full weight lets it
compete with labels that are 25x tighter.

Two guards, both deliberate:

* **The floor.** 7,088 of 7,367 rows are single measurements with ``std == 0``.
  Unfloored, ``1/std`` is infinite for every one of them, which is not
  "confident" -- it is "unmeasured". The floor defaults to the observed median
  spread, 5.6 K, so a single measurement is treated as typical rather than
  perfect.
* **Renormalisation.** Weights are rescaled to mean 1.0, so switching this on
  does not also change the effective learning rate. Without it, arm A4 would
  confound label weighting with an LR change and its result would be
  unattributable.

``reliability == red`` rows (4 of them) are dropped, and the dropped rows are
returned rather than discarded so the caller can log what left.

Torch-free.
"""

from __future__ import annotations

from collections.abc import Sequence

from polyt5.data.tg_metadata import TgExample
from polyt5.utils import get_logger

__all__ = [
    "DEFAULT_STD_FLOOR",
    "RED_RELIABILITY",
    "drop_red_reliability",
    "reliability_weights",
]

_logger = get_logger("polyt5.data.weighting")

#: Curator flag marking a measurement the LamaLab curation distrusts.
RED_RELIABILITY = "red"

#: Median experimental spread over the 279 repeatedly measured polymers (K).
#: Used as the weight floor: a single measurement is typical, not perfect.
DEFAULT_STD_FLOOR = 5.6


def drop_red_reliability(
    examples: Sequence[TgExample],
) -> tuple[list[TgExample], list[TgExample]]:
    """Split examples into those to train on and those flagged ``red``.

    Args:
        examples: Prepared examples.

    Returns:
        ``(kept, dropped)``, both in input order. ``dropped`` is returned
        rather than thrown away so the caller can record what left and why.
    """
    kept: list[TgExample] = []
    dropped: list[TgExample] = []
    for example in examples:
        (dropped if example.row.reliability == RED_RELIABILITY else kept).append(example)
    if dropped:
        _logger.info(
            "drop_red_reliability: dropped %d of %d rows flagged %r",
            len(dropped),
            len(examples),
            RED_RELIABILITY,
        )
    return kept, dropped


def reliability_weights(
    examples: Sequence[TgExample],
    *,
    floor: float = DEFAULT_STD_FLOOR,
    normalize: bool = True,
) -> list[float]:
    """Weight each example by ``1 / max(std, floor)``.

    Args:
        examples: Prepared examples, already filtered by
            :func:`drop_red_reliability`.
        floor: Lower bound on the standard deviation. Must be positive.
        normalize: Rescale so the weights average exactly 1.0, keeping the
            effective learning rate unchanged when weighting is switched on.

    Returns:
        One weight per example, in input order. Empty input gives empty output.

    Raises:
        ValueError: If ``floor`` is not strictly positive -- an unfloored
            ``std == 0`` row would acquire infinite weight.
    """
    if floor <= 0.0:
        raise ValueError(
            f"floor must be > 0, got {floor}: without a positive floor a single-measurement "
            "polymer (std = 0) acquires infinite weight and dominates every gradient"
        )
    raw = [1.0 / max(float(example.row.std), floor) for example in examples]
    if not raw or not normalize:
        return raw
    mean = sum(raw) / len(raw)
    return [weight / mean for weight in raw]
