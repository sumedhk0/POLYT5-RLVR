"""Per-column standardisation fitted on the train split only.

Raw Kelvin (mu ~ 417, sigma ~ 113) makes the regression loss landscape awkward,
and the 100 LamaLab descriptors span wildly different scales. Both are handled
by the same object: the Tg target is a one-column matrix.

Two rules are load-bearing:

* **Train statistics only.** ``fit`` is called on the train rows of one split;
  ``transform`` is then applied to val and test with those same numbers.
  Fitting on all rows would leak the test set's location and scale.
* **Drop, log, never impute.** A column that is constant or carries any
  non-finite value on the fitting rows is removed and named in the log and in
  :attr:`Standardizer.dropped`. Filling it with a mean would invent
  measurements the CSV does not contain.

Torch-free: numpy only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from polyt5.utils import get_logger

__all__ = ["Standardizer", "fit_target_standardizer"]

_logger = get_logger("polyt5.data.standardize")


@dataclass(frozen=True)
class Standardizer:
    """Column means and standard deviations, plus the columns that were dropped.

    Attributes:
        columns: Names of the kept columns, in output order.
        keep_index: Positions of those columns in the ORIGINAL input matrix, so
            :meth:`transform` can select them from a full-width array.
        mean: Per-kept-column mean from the fitting rows.
        std: Per-kept-column standard deviation from the fitting rows; every
            entry is ``> min_std`` by construction, so division is safe.
        dropped: Names of columns removed as constant or non-finite.
    """

    columns: tuple[str, ...]
    keep_index: tuple[int, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    dropped: tuple[str, ...]

    @property
    def n_features(self) -> int:
        """How many columns survive standardisation."""
        return len(self.columns)

    @classmethod
    def fit(
        cls, values: np.ndarray, columns: Sequence[str], *, min_std: float = 1e-8
    ) -> Standardizer:
        """Fit on ``values``, dropping columns that carry no usable signal.

        Args:
            values: ``(n_rows, n_columns)`` array of the FITTING rows only.
            columns: One name per column of ``values``.
            min_std: Standard deviations at or below this count as constant.

        Returns:
            A fitted :class:`Standardizer`.

        Raises:
            ValueError: If ``values`` is not 2-D, or its width does not match
                ``columns``.
        """
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"values must be 2-D (n_rows, n_columns), got shape {array.shape}")
        if array.shape[1] != len(columns):
            raise ValueError(
                f"values has {array.shape[1]} columns but {len(columns)} column names were given"
            )

        finite = np.isfinite(array).all(axis=0)
        # Compute the std only where every value is finite: a column with a NaN
        # would otherwise produce a NaN std and a RuntimeWarning on comparison.
        std = np.zeros(array.shape[1], dtype=np.float64)
        if finite.any():
            std[finite] = array[:, finite].std(axis=0, ddof=0)
        keep = finite & (std > min_std)

        dropped = tuple(name for name, kept in zip(columns, keep, strict=True) if not kept)
        if dropped:
            _logger.info(
                "Standardizer.fit: dropped %d of %d columns as constant or non-finite on the "
                "fitting rows (never imputed): %s",
                len(dropped),
                len(columns),
                ", ".join(dropped),
            )
        keep_index = tuple(int(i) for i in np.flatnonzero(keep))
        return cls(
            columns=tuple(name for name, kept in zip(columns, keep, strict=True) if kept),
            keep_index=keep_index,
            mean=tuple(float(v) for v in array[:, keep].mean(axis=0)),
            std=tuple(float(v) for v in std[keep]),
            dropped=dropped,
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Select the kept columns of ``values`` and standardise them.

        Args:
            values: ``(n_rows, n_original_columns)`` array.

        Returns:
            ``(n_rows, n_features)`` standardised array.
        """
        array = np.asarray(values, dtype=np.float64)
        selected = array[:, list(self.keep_index)]
        return (selected - np.asarray(self.mean)) / np.asarray(self.std)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Map standardised values back to their original units.

        Args:
            values: ``(n_rows, n_features)`` standardised array.

        Returns:
            ``(n_rows, n_features)`` array in original units.
        """
        array = np.asarray(values, dtype=np.float64)
        return array * np.asarray(self.std) + np.asarray(self.mean)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "columns": list(self.columns),
            "keep_index": list(self.keep_index),
            "mean": list(self.mean),
            "std": list(self.std),
            "dropped": list(self.dropped),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Standardizer:
        """Rebuild a standardizer from :meth:`to_dict` output."""
        return cls(
            columns=tuple(payload["columns"]),
            keep_index=tuple(int(i) for i in payload["keep_index"]),
            mean=tuple(float(v) for v in payload["mean"]),
            std=tuple(float(v) for v in payload["std"]),
            dropped=tuple(payload.get("dropped", ())),
        )


def fit_target_standardizer(values: Sequence[float], *, name: str = "Tg") -> Standardizer:
    """Fit the scalar property standardizer on one split's train targets.

    Args:
        values: Train-split property values in their natural units (Kelvin).
        name: Column name recorded in the standardizer.

    Returns:
        A one-column :class:`Standardizer`.

    Raises:
        ValueError: If the targets have no variance, which would make the
            inverse transform meaningless and the forward one a division by
            zero.
    """
    standardizer = Standardizer.fit(np.asarray(values, dtype=np.float64)[:, None], [name])
    if standardizer.n_features == 0:
        raise ValueError(
            f"no usable variance in {len(values)} target values: a constant target cannot be "
            "standardised, and a model trained on it would predict one number forever"
        )
    return standardizer
