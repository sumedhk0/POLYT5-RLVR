"""Property-prediction metrics for a seq2seq regressor that emits text.

polyT5 predicts properties by *generating* them: beam search (width 4) produces
a string, which is "decoded into floating-point numbers, and filtered to remove
any invalid or non-numeric outputs". That filtering step is a real part of the
evaluation, not a preprocessing detail -- a model that emits ``"soluble"``
instead of a number has failed on that example, and hiding those examples by
quietly shrinking the denominator would flatter the reported MAE.

So :func:`regression_report` always records how many outputs were non-numeric
alongside the metrics computed on the ones that survived. The paper reports MAE
as its primary metric, plus RMSE, R^2 and Pearson r, averaged over five random
splits -- see :func:`aggregate_over_splits`.

Degenerate cases return ``None`` for the affected metric rather than ``NaN`` or
an exception: R^2 and Pearson r are genuinely undefined when the targets have
zero variance, and a ``None`` in a metrics file is honest where a ``NaN`` is a
bug waiting to be averaged.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = [
    "METRIC_NAMES",
    "RegressionReport",
    "aggregate_over_splits",
    "parse_numeric_predictions",
    "regression_report",
]

#: Metrics carried through :func:`aggregate_over_splits`.
METRIC_NAMES: tuple[str, ...] = ("mae", "rmse", "r2", "pearson_r", "non_numeric_rate")

# A complete decimal or scientific-notation number and nothing else. Deliberately
# strict: it rejects "1.2.3", "412.5 K", "1,234.5", and the float() specials
# "nan"/"inf", all of which are model failures rather than numbers.
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


@dataclass(frozen=True)
class RegressionReport:
    """Property-prediction metrics for one evaluation set.

    Attributes:
        n_total: Number of examples evaluated.
        n_valid_numeric: Predictions that parsed as a number.
        n_non_numeric: Predictions that did not. Reported explicitly so a low
            MAE on a small surviving subset cannot be mistaken for a good model.
        non_numeric_rate: ``n_non_numeric / n_total``, or ``0.0`` when empty.
        mae: Mean absolute error, the paper's primary metric. ``None`` when
            there is nothing to score.
        rmse: Root mean squared error, or ``None``.
        r2: Coefficient of determination, or ``None`` when it is undefined
            (fewer than two pairs, or zero variance in the targets).
        pearson_r: Pearson correlation, or ``None`` when it is undefined
            (fewer than two pairs, or zero variance on either side).
    """

    n_total: int
    n_valid_numeric: int
    n_non_numeric: int
    non_numeric_rate: float
    mae: float | None
    rmse: float | None
    r2: float | None
    pearson_r: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return asdict(self)


def parse_numeric_predictions(texts: Sequence[Any]) -> tuple[list[float | None], int]:
    """Decode generated strings into floats, flagging the ones that are not numbers.

    Implements the paper's "decoded into floating-point numbers, and filtered to
    remove any invalid or non-numeric outputs" step, but *records* the removals
    instead of discarding them silently.

    Args:
        texts: Raw decoded model outputs. Values that are already ``int`` or
            ``float`` pass through, so a caller that has already parsed can use
            this function too.

    Returns:
        A ``(values, n_non_numeric)`` pair. ``values`` is positionally aligned
        with ``texts``, holding a float where parsing succeeded and ``None``
        where it failed.

    Note:
        Parsing is strict: surrounding whitespace is stripped, but a unit suffix
        (``"412.5 K"``), a thousands separator (``"1,234.5"``), or a malformed
        number (``"1.2.3"``) is a non-numeric output, as are the ``float()``
        specials ``"nan"`` and ``"inf"``, which are not measurements.
        # [AMBIGUITY] the paper does not define what counts as "numeric"; the
        # strict reading is used because a lenient one would invent a value the
        # model did not emit.
    """
    values: list[float | None] = []
    n_non_numeric = 0

    for text in texts:
        value = _parse_one(text)
        if value is None:
            n_non_numeric += 1
        values.append(value)
    return values, n_non_numeric


def _parse_one(text: Any) -> float | None:
    """Parse a single decoded output into a finite float, or ``None``."""
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        return float(text) if math.isfinite(float(text)) else None
    if not isinstance(text, str):
        return None

    stripped = text.strip()
    if not _NUMBER_RE.match(stripped):
        return None
    try:
        value = float(stripped)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def regression_report(
    y_true: Sequence[Any],
    y_pred_text: Sequence[Any],
    *,
    drop_non_numeric: bool = True,
) -> RegressionReport:
    """Score generated property predictions against ground truth.

    Args:
        y_true: Ground-truth property values, one per example.
        y_pred_text: Decoded model outputs, positionally aligned with ``y_true``.
        drop_non_numeric: When ``True`` (the paper's protocol), metrics are
            computed on the pairs whose prediction parsed as a number, and the
            discarded count is still reported. When ``False``, the presence of
            any non-numeric output makes every metric ``None`` -- for callers
            who would rather have no number than a number over a subset.

    Returns:
        A :class:`RegressionReport`.

    Raises:
        ValueError: If the two sequences have different lengths. That is a
            caller bug, not adversarial model output, so it is not swallowed.
    """
    if len(y_true) != len(y_pred_text):
        raise ValueError(
            f"y_true and y_pred_text must be the same length, "
            f"got {len(y_true)} and {len(y_pred_text)}"
        )

    n_total = len(y_true)
    parsed, n_non_numeric = parse_numeric_predictions(y_pred_text)

    pairs: list[tuple[float, float]] = []
    for truth, prediction in zip(y_true, parsed, strict=True):
        if prediction is None:
            continue
        truth_value = _parse_one(truth)
        if truth_value is None:
            # A missing/unusable ground-truth value cannot be scored either.
            continue
        pairs.append((truth_value, prediction))

    n_valid_numeric = n_total - n_non_numeric
    non_numeric_rate = 0.0 if n_total == 0 else n_non_numeric / n_total

    empty = RegressionReport(
        n_total=n_total,
        n_valid_numeric=n_valid_numeric,
        n_non_numeric=n_non_numeric,
        non_numeric_rate=non_numeric_rate,
        mae=None,
        rmse=None,
        r2=None,
        pearson_r=None,
    )
    if not pairs:
        return empty
    if not drop_non_numeric and n_non_numeric > 0:
        return empty

    truth_array = np.asarray([p[0] for p in pairs], dtype=float)
    pred_array = np.asarray([p[1] for p in pairs], dtype=float)
    residuals = truth_array - pred_array

    mae = float(np.abs(residuals).mean())
    rmse = float(math.sqrt(float((residuals**2).mean())))

    r2: float | None = None
    pearson: float | None = None
    if len(pairs) >= 2:
        ss_tot = float(((truth_array - truth_array.mean()) ** 2).sum())
        if ss_tot > 0.0:
            ss_res = float((residuals**2).sum())
            r2 = 1.0 - ss_res / ss_tot
            if float(pred_array.std()) > 0.0:
                matrix = np.corrcoef(truth_array, pred_array)
                candidate = float(matrix[0, 1])
                pearson = candidate if math.isfinite(candidate) else None

    return RegressionReport(
        n_total=n_total,
        n_valid_numeric=n_valid_numeric,
        n_non_numeric=n_non_numeric,
        non_numeric_rate=non_numeric_rate,
        mae=mae,
        rmse=rmse,
        r2=r2,
        pearson_r=pearson,
    )


def aggregate_over_splits(
    reports: Sequence[RegressionReport], *, ddof: int = 0
) -> dict[str, Any]:
    """Average metrics across repeated splits, as the paper reports them.

    The paper's headline numbers (e.g. Tg RMSE 40.82, R^2 0.86, r 0.93 for
    polyT5-medium) are averages over five random splits.

    Args:
        reports: One :class:`RegressionReport` per split.
        ddof: Delta degrees of freedom for the standard deviation. ``0`` is the
            population standard deviation over the splits actually run; pass
            ``1`` for the sample standard deviation.
            # [AMBIGUITY] the paper does not state which convention its spread
            # figures use, so the choice is left to the caller.

    Returns:
        ``{"n_splits": int, "<metric>": {"mean": float|None, "std": float|None,
        "n": int}}`` for each metric in :data:`METRIC_NAMES`. A metric that was
        ``None`` in every split yields ``None`` for mean and std with ``n=0``;
        splits where it was ``None`` are excluded rather than treated as zero.
    """
    out: dict[str, Any] = {"n_splits": len(reports)}

    for name in METRIC_NAMES:
        values = [
            float(getattr(report, name))
            for report in reports
            if getattr(report, name) is not None
        ]
        if not values:
            out[name] = {"mean": None, "std": None, "n": 0}
            continue
        array = np.asarray(values, dtype=float)
        std = float(array.std(ddof=ddof)) if len(values) > ddof else None
        out[name] = {"mean": float(array.mean()), "std": std, "n": len(values)}
    return out
