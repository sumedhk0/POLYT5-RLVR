"""Did a shared-encoder arm cost us generation quality?

The multi-task arms give one encoder two objectives, and a gain on prediction
may be paid for out of generation. Spec section 6 requires the trade to be
SURFACED, not absorbed: "a prediction gain bought with a generation loss is a
trade to surface, not a success to report".

The reference rates are read from ``artifacts/baseline/frozen_baseline.json``
under ``arm_b_tuned_sampling`` -- never hard-coded here -- so this check cannot
drift away from the artifact it claims to compare against. Arm B is the tuned
sampling point (temperature 0.7, top_p 0.95); comparing against a different
sampling point would make the row incomparable on every column.

An unmeasured TP rate stays ``None``. It is not a failed TP, and it must not
be averaged as a zero.

Torch-free, like the rest of :mod:`polyt5.evaluation`. This module must not
import :mod:`polyt5.training`: evaluation reads results, it does not depend on
the thing that produced them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polyt5.evaluation.generation_metrics import GenerationReport

__all__ = [
    "DEFAULT_REGRESSION_TOLERANCE",
    "GenerationRegressionCheck",
    "check_generation_regression",
    "load_arm_b_generation_baseline",
    "requires_generation_check",
]

#: How far below the frozen rate an arm may land before it counts as a
#: regression. Two percentage points; the frozen rates are quoted to one
#: decimal place (55.8%, 58.8%), so a tighter band would be reading noise.
DEFAULT_REGRESSION_TOLERANCE = 0.02

_ARM_B_KEY = "arm_b_tuned_sampling"


@dataclass(frozen=True)
class GenerationRegressionCheck:
    """One arm's generation rates against the frozen Arm B rates."""

    arm: str
    pv_rate: float
    tp_rate: float | None
    baseline_pv_rate: float
    baseline_tp_rate: float
    pv_delta: float
    tp_delta: float | None
    tolerance: float
    regressed: bool
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "arm": self.arm,
            "pv_rate": self.pv_rate,
            "tp_rate": self.tp_rate,
            "baseline_pv_rate": self.baseline_pv_rate,
            "baseline_tp_rate": self.baseline_tp_rate,
            "pv_delta": self.pv_delta,
            "tp_delta": self.tp_delta,
            "tolerance": self.tolerance,
            "regressed": self.regressed,
            "verdict": self.verdict,
        }


def load_arm_b_generation_baseline(path: str | Path) -> tuple[float, float]:
    """Read Arm B's frozen PV and TP rates.

    Args:
        path: Path to ``artifacts/baseline/frozen_baseline.json``.

    Returns:
        ``(pv_rate, tp_rate)``.

    Raises:
        ValueError: If the artifact carries no ``arm_b_tuned_sampling`` with
            both rates.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    arm_b = payload.get(_ARM_B_KEY) or {}
    pv_rate, tp_rate = arm_b.get("pv_rate"), arm_b.get("tp_rate")
    if pv_rate is None or tp_rate is None:
        raise ValueError(
            f"{path} carries no {_ARM_B_KEY} pv_rate/tp_rate; the generation comparison "
            "point must come from the frozen artifact, not from a constant"
        )
    return float(pv_rate), float(tp_rate)


def requires_generation_check(switches: dict[str, bool]) -> bool:
    """Whether a configuration needs the generation regression check.

    Only a configuration that trains a second objective through the shared
    encoder can trade generation away, so this is the ``multitask`` switch.

    Args:
        switches: The configuration's five switches.

    Returns:
        ``True`` when generation must be re-measured for this configuration.
    """
    return bool(switches.get("multitask", False))


def check_generation_regression(
    report: GenerationReport,
    *,
    arm: str,
    baseline_pv_rate: float,
    baseline_tp_rate: float,
    tolerance: float = DEFAULT_REGRESSION_TOLERANCE,
) -> GenerationRegressionCheck:
    """Compare one arm's generation rates against Arm B's frozen rates.

    Args:
        report: The arm's generation evaluation, sampled at Arm B's
            temperature and top_p.
        arm: The configuration id, for the record.
        baseline_pv_rate: Arm B's PV rate from the frozen artifact.
        baseline_tp_rate: Arm B's TP rate from the frozen artifact.
        tolerance: How far below either rate is still acceptable.

    Returns:
        A :class:`GenerationRegressionCheck`.

    Raises:
        ValueError: If ``tolerance`` is not positive -- a zero tolerance would
            make sampling noise read as a regression.
    """
    if tolerance <= 0.0:
        raise ValueError(
            f"tolerance must be > 0, got {tolerance}: with no band, sampling noise reads "
            "as a regression and the check becomes uninformative"
        )
    pv_rate = report.counts.pv_rate
    tp_rate = report.target_property_rate
    pv_delta = pv_rate - baseline_pv_rate
    tp_delta = None if tp_rate is None else tp_rate - baseline_tp_rate

    regressed = pv_delta < -tolerance or (tp_delta is not None and tp_delta < -tolerance)
    if regressed:
        verdict = "regressed — a prediction gain bought here is a trade, not a success"
    elif pv_delta > tolerance and (tp_delta is None or tp_delta > -tolerance):
        verdict = "improved"
    else:
        verdict = "no regression"

    return GenerationRegressionCheck(
        arm=arm,
        pv_rate=pv_rate,
        tp_rate=tp_rate,
        baseline_pv_rate=baseline_pv_rate,
        baseline_tp_rate=baseline_tp_rate,
        pv_delta=pv_delta,
        tp_delta=tp_delta,
        tolerance=tolerance,
        regressed=regressed,
        verdict=verdict,
    )
