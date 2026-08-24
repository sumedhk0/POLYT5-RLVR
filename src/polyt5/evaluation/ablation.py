"""The Group A ablation matrix and its pre-registered verdict.

The criterion was fixed before any run (spec section 6). A configuration
**helps** only if its five-split mean MAE is below the frozen baseline's mean
minus one standard deviation -- 28.6733 - 0.7591 = 27.9142 K. A configuration
landing inside that spread is **no effect**, not a small win, and one above it
**hurts**.

Reporting discipline (spec section 9). Three claims never merge:

* *the paper reports* -- Tg prediction on 5,130 withheld labels;
* *our reproduction obtains* -- 28.6733 +/- 0.7591 K on 7,367 LamaLab labels;
* *our extension obtains* -- Group A.

Group A results are the third category and are stamped as such. An unrun
configuration renders as ``not run`` with ``None`` metrics: an empty cell must
never be mistakable for a zero. Rows are emitted in the order they are given
(arm order), never sorted into a leaderboard, so a configuration that hurts is
as prominent as one that helps.

Torch-free, like the rest of :mod:`polyt5.evaluation`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from polyt5.evaluation.regression_metrics import RegressionReport, aggregate_over_splits

__all__ = [
    "BASELINE_ARM",
    "CLAIM_CATEGORY",
    "CLAIM_NOTE",
    "EFFECT_HELPS",
    "EFFECT_HURTS",
    "EFFECT_NOT_RUN",
    "EFFECT_NO_EFFECT",
    "ArmResult",
    "build_ablation_matrix",
    "classify_effect",
    "format_ablation_matrix",
    "success_threshold",
]

#: The configuration every other one is compared against.
BASELINE_ARM = "B0"

EFFECT_HELPS = "helps"
EFFECT_NO_EFFECT = "no effect"
EFFECT_HURTS = "hurts"
EFFECT_NOT_RUN = "not run"

#: Which of the three never-merged claims a Group A number is.
CLAIM_CATEGORY = "our extension obtains"

CLAIM_NOTE = (
    "Group A numbers are our extension's improvements to OUR OWN reproduction. They are "
    "not comparable to the paper's reported Tg figures, which are a different quantity "
    "(5,130 withheld labels) measured on a different dataset, and must never be presented "
    "as closing a gap to the paper."
)


def success_threshold(baseline_mean: float, baseline_std: float) -> float:
    """The pre-registered MAE a configuration must beat to count as helping.

    Args:
        baseline_mean: Frozen baseline five-split mean MAE (K).
        baseline_std: Its standard deviation over those splits (K).

    Returns:
        ``baseline_mean - baseline_std``.
    """
    return baseline_mean - baseline_std


def classify_effect(
    mae_mean: float | None, *, baseline_mean: float, baseline_std: float
) -> str:
    """Apply the pre-registered verdict to one configuration's mean MAE.

    Args:
        mae_mean: The configuration's five-split mean MAE, or ``None`` when it
            has not been run.
        baseline_mean: Frozen baseline mean MAE.
        baseline_std: Frozen baseline standard deviation.

    Returns:
        One of :data:`EFFECT_HELPS`, :data:`EFFECT_NO_EFFECT`,
        :data:`EFFECT_HURTS`, :data:`EFFECT_NOT_RUN`.
    """
    if mae_mean is None:
        return EFFECT_NOT_RUN
    if mae_mean < success_threshold(baseline_mean, baseline_std):
        return EFFECT_HELPS
    if mae_mean > baseline_mean + baseline_std:
        return EFFECT_HURTS
    return EFFECT_NO_EFFECT


@dataclass(frozen=True)
class ArmResult:
    """One configuration's aggregated five-split metrics."""

    arm: str
    switches: dict[str, bool]
    n_splits: int
    mae_mean: float | None
    mae_std: float | None
    rmse_mean: float | None
    rmse_std: float | None
    r2_mean: float | None
    r2_std: float | None
    non_numeric_rate_mean: float | None

    @classmethod
    def from_reports(
        cls, arm: str, switches: dict[str, bool], reports: Sequence[RegressionReport]
    ) -> ArmResult:
        """Aggregate one configuration's per-split reports.

        Args:
            arm: Configuration id.
            switches: Which of the five changes this configuration turns on.
            reports: One report per completed split; an empty sequence yields
                an all-``None`` result rather than zeros.

        Returns:
            An :class:`ArmResult`.
        """
        aggregate = aggregate_over_splits(reports)

        def value(metric: str, field: str) -> float | None:
            entry = aggregate.get(metric) or {}
            return entry.get(field)

        return cls(
            arm=arm,
            switches=dict(switches),
            n_splits=len(reports),
            mae_mean=value("mae", "mean"),
            mae_std=value("mae", "std"),
            rmse_mean=value("rmse", "mean"),
            rmse_std=value("rmse", "std"),
            r2_mean=value("r2", "mean"),
            r2_std=value("r2", "std"),
            non_numeric_rate_mean=value("non_numeric_rate", "mean"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "arm": self.arm,
            "switches": dict(self.switches),
            "n_splits": self.n_splits,
            "mae_mean": self.mae_mean,
            "mae_std": self.mae_std,
            "rmse_mean": self.rmse_mean,
            "rmse_std": self.rmse_std,
            "r2_mean": self.r2_mean,
            "r2_std": self.r2_std,
            "non_numeric_rate_mean": self.non_numeric_rate_mean,
        }


def build_ablation_matrix(
    results: Sequence[ArmResult], *, baseline_mean: float, baseline_std: float
) -> dict[str, Any]:
    """Assemble the arm x metric matrix with its verdicts and provenance.

    Args:
        results: One :class:`ArmResult` per configuration, in the order they
            should be reported.
        baseline_mean: Frozen baseline five-split mean MAE.
        baseline_std: Its standard deviation.

    Returns:
        A JSON-serialisable matrix.

    Raises:
        ValueError: If an arm appears twice, or ``baseline_std`` is not
            positive (which would collapse the verdict to a point comparison).
    """
    if baseline_std <= 0.0:
        raise ValueError(
            f"baseline_std must be > 0, got {baseline_std}: without a spread there is no "
            "'no effect' band and every difference would read as a result"
        )
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.arm in seen:
            raise ValueError(f"duplicate arm {result.arm!r} in the ablation matrix")
        seen.add(result.arm)
        rows.append(
            {
                **result.to_dict(),
                "effect": classify_effect(
                    result.mae_mean, baseline_mean=baseline_mean, baseline_std=baseline_std
                ),
                "mae_delta_vs_baseline": (
                    None if result.mae_mean is None else result.mae_mean - baseline_mean
                ),
            }
        )
    return {
        "claim_category": CLAIM_CATEGORY,
        "claim_note": CLAIM_NOTE,
        "baseline_arm": BASELINE_ARM,
        "baseline_mae_mean": baseline_mean,
        "baseline_mae_std": baseline_std,
        "success_threshold": success_threshold(baseline_mean, baseline_std),
        "rows": rows,
    }


def _cell(row: dict[str, Any], key: str, width: int, digits: int) -> str:
    """Render one numeric cell, or right-aligned ``n/a`` when it is ``None``."""
    value = row.get(key)
    return f"{'n/a':>{width}}" if value is None else f"{value:>{width}.{digits}f}"


def format_ablation_matrix(matrix: dict[str, Any]) -> str:
    """Render the matrix as a fixed-width table.

    Args:
        matrix: Output of :func:`build_ablation_matrix`.

    Returns:
        The table as a string. The caller prints it; this module never does.
    """
    header = (
        f"{'arm':<5}{'MAE':>10}{'+/-':>8}{'RMSE':>10}{'R2':>8}"
        f"{'non_numeric':>13}{'splits':>8}  verdict"
    )
    lines = [
        f"Group A ablation -- {matrix['claim_category']}",
        (
            f"baseline {matrix['baseline_arm']} = {matrix['baseline_mae_mean']:.4f} "
            f"+/- {matrix['baseline_mae_std']:.4f} K; "
            f"helps below {matrix['success_threshold']:.4f} K"
        ),
        "=" * len(header),
        header,
        "-" * len(header),
    ]
    for row in matrix["rows"]:
        lines.append(
            f"{row['arm']:<5}{_cell(row, 'mae_mean', 10, 4)}{_cell(row, 'mae_std', 8, 4)}"
            f"{_cell(row, 'rmse_mean', 10, 4)}{_cell(row, 'r2_mean', 8, 4)}"
            f"{_cell(row, 'non_numeric_rate_mean', 13, 4)}{row['n_splits']:>8}  "
            f"{row['effect']}"
        )
    lines.extend(["-" * len(header), matrix["claim_note"]])
    return "\n".join(lines)
