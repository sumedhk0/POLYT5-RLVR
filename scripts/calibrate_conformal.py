"""Calibrate split-conformal intervals for a trained regression predictor.

Phase 4 Group B. NOT part of the published polyT5 method -- see
``docs/superpowers/specs/2026-08-25-phase4-group-b-conformal-design.md``.

Reads a run's ``predictions.jsonl``, splits the held-out set into DISJOINT
calibration and validation halves, fits the conformal half-width on the first and
reports realised coverage on the second. Coverage measured on the calibration set
would be the tautology ``rank / n`` -- true for any model, even a random one -- so
the split is what makes the reported number mean anything.

Usage:
    python scripts/calibrate_conformal.py \
        --predictions results/a1_full_corpus/A1/split_0_*/predictions.jsonl \
        --out artifacts/conformal/a1_full_corpus.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polyt5.evaluation.conformal import (  # noqa: E402
    ConformalRegressor,
    split_calibration,
)
from polyt5.utils import get_logger  # noqa: E402

DEFAULT_ALPHAS = (0.20, 0.10, 0.05)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--alphas", nargs="*", type=float, default=list(DEFAULT_ALPHAS),
        help="Miscoverage rates. Default 0.20 0.10 0.05 (80%%, 90%%, 95%% intervals).",
    )
    p.add_argument(
        "--calibration-fraction", type=float, default=0.5,
        help="Share of the held-out set used to fit the half-width. Default 0.5.",
    )
    p.add_argument("--seed", type=int, default=20260825)
    p.add_argument(
        "--repeats", type=int, default=200,
        help="Independent calibration/validation partitions to average coverage over.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = get_logger("polyt5.calibrate_conformal")

    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines()]
    pairs = []
    n_non_numeric = 0
    for row in rows:
        try:
            pairs.append((float(row["prediction"]), float(row["target"])))
        except (TypeError, ValueError):
            n_non_numeric += 1
    if n_non_numeric:
        # A regression head cannot produce these; a text head can. Dropping them
        # silently would quietly narrow the interval on the arm that needs it most.
        logger.warning(
            "dropped %d non-numeric prediction(s): conformal cannot bound an interval "
            "around a decode that is not a number", n_non_numeric
        )
    n = len(pairs)
    n_cal = int(round(n * args.calibration_fraction))
    cal_idx, val_idx = split_calibration(n, n_cal, seed=args.seed)

    cal_p = [pairs[i][0] for i in cal_idx]
    cal_t = [pairs[i][1] for i in cal_idx]
    val_p = [pairs[i][0] for i in val_idx]
    val_t = [pairs[i][1] for i in val_idx]

    out: dict = {
        "source_predictions": str(args.predictions),
        "n_held_out": n,
        "n_calibration": len(cal_idx),
        "n_validation": len(val_idx),
        "n_non_numeric_dropped": n_non_numeric,
        "seed": args.seed,
        "note": (
            "Split conformal. Coverage is measured on a DISJOINT half; on the "
            "calibration half it would be the tautology rank/n. The guarantee is valid "
            "at any n, but with this n the coverage ESTIMATE is imprecise -- read the "
            "CI, not the point estimate."
        ),
        "levels": [],
    }
    logger.info("%d held out -> %d calibration / %d validation", n, len(cal_idx), len(val_idx))
    for alpha in sorted(args.alphas):
        model = ConformalRegressor.calibrate(cal_p, cal_t, alpha=alpha)
        report = model.coverage(val_p, val_t)
        out["levels"].append({
            "alpha": alpha,
            "target_coverage": report.target_coverage,
            "half_width_k": model.half_width,
            "interval_width_k": report.mean_width,
            "measured_coverage": report.coverage,
            "coverage_ci95": list(report.coverage_ci),
            "n_validation": report.n,
            "covers_target": (
                report.coverage_ci[0] <= report.target_coverage <= report.coverage_ci[1]
            ),
        })
        logger.info(
            "alpha=%.2f  +/- %6.2f K  measured coverage %.3f  CI [%.3f, %.3f]  target %.2f",
            alpha, model.half_width, report.coverage,
            report.coverage_ci[0], report.coverage_ci[1], report.target_coverage,
        )

    # One split gives a noisy coverage estimate. Nothing retrains, so repeat the
    # split many times and average -- this measures the METHOD's coverage at this
    # calibration size rather than one lucky or unlucky partition.
    out["repeated_split"] = {
        "n_repeats": args.repeats,
        "note": (
            "Mean realised coverage over independent calibration/validation partitions "
            "of the same held-out set. Addresses the noise in a single split; does NOT "
            "address the underlying held-out set being small."
        ),
        "levels": [],
    }
    for alpha in sorted(args.alphas):
        covs, widths = [], []
        for r in range(args.repeats):
            c_idx, v_idx = split_calibration(n, n_cal, seed=args.seed + 1000 * (r + 1))
            m = ConformalRegressor.calibrate(
                [pairs[i][0] for i in c_idx], [pairs[i][1] for i in c_idx], alpha=alpha
            )
            rep = m.coverage([pairs[i][0] for i in v_idx], [pairs[i][1] for i in v_idx])
            covs.append(rep.coverage)
            widths.append(m.half_width)
        mean_cov = sum(covs) / len(covs)
        mean_hw = sum(widths) / len(widths)
        out["repeated_split"]["levels"].append({
            "alpha": alpha,
            "target_coverage": 1.0 - alpha,
            "mean_coverage": mean_cov,
            "mean_half_width_k": mean_hw,
            "coverage_p05": sorted(covs)[int(0.05 * len(covs))],
            "coverage_p95": sorted(covs)[int(0.95 * len(covs)) - 1],
            "shortfall": (1.0 - alpha) - mean_cov,
        })
        logger.info(
            "alpha=%.2f  over %d splits: mean coverage %.3f (target %.2f, shortfall %+.3f)"
            "  mean half-width %.2f K",
            alpha, args.repeats, mean_cov, 1.0 - alpha, (1.0 - alpha) - mean_cov, mean_hw,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
