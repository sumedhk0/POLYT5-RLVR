"""Arm-comparison matrix: Arm A/B (Phase 1/2, supervised) vs the four Phase-3
GRPO/RLVR arms (Task 6's ``accuracy``/``validity``/``composite``/``constraint``).

Phase 3. NOT part of the published polyT5 method - see ``docs/rlvr_plan.md``.

For every arm this script SAMPLES FRESH candidates under the fixed evaluation
protocol recorded in ``artifacts/baseline/frozen_baseline.json``
(``evaluation_protocol``: targets 300/400/500 K, ``n_per_target`` each) and
scores them -- it never reuses the numbers already recorded there for Arm A/B.
That keeps every row of the matrix produced by the SAME measurement, which is
the only way the columns are actually comparable:

* **Arm A / Arm B** -- the frozen ``generation`` checkpoint (SHA-256 verified),
  resampled at the ``arm_a_default_sampling`` / ``arm_b_tuned_sampling``
  temperature and ``top_p`` recorded in the frozen baseline.
* **The four RLVR arms** -- each arm's own GRPO-trained policy checkpoint
  (latest ``step_*.pt`` under ``results/grpo_<arm>/checkpoints/``, written by
  ``scripts/train_grpo.py``), resampled at that arm's own config's rollout
  temperature/``top_p``. An arm with no run directory yet is skipped with a
  warning rather than aborting the whole comparison -- this script is meant to
  run again as each arm finishes training, not only once everything is done.

Every row is scored TWICE:

* by **the auditor** (``frozen_baseline.json``'s ``auditor`` split,
  ``tg_predictor_split4``) -- held out of every reward path, so this number
  cannot be an artifact of the reward model an RLVR arm was optimised against.
* by **the reward ensemble** (the same four splits ``scripts/train_grpo.py``
  builds the reward from) -- this is literally "the metric it optimized" for
  the RLVR arms, and gives a same-scale baseline for Arm A/B too.

The pre-registered ``success_criterion`` -- "An RLVR arm succeeds only if it
beats arm_b on the metric it optimized AND the gain survives scoring by the
auditor" -- is applied per RLVR arm as a two-clause test:

    beats_arm_b   = ensemble-scored optimized metric beats arm_b's
                    ensemble-scored value of the SAME metric
    survives_audit = auditor-scored optimized metric beats arm_b's
                    auditor-scored value of the SAME metric
    success        = beats_arm_b AND survives_audit

[AMBIGUITY] The frozen baseline names "the metric it optimized" without
pinning one column per arm, and the four arms do not optimize the same axis;
see ``ARM_METRIC`` below for the mapping used and its rationale.

Outputs (always ``results/arm_comparison/`` unless ``--out`` overrides it):
    matrix.csv     one row per arm, every measured column
    matrix.md      the same table, human-readable
    summary.json   the full row data plus the success verdict per RLVR arm

Usage:
    python scripts/compare_arms.py
    python scripts/compare_arms.py --arms accuracy validity   # partial run
    python scripts/compare_arms.py --n-per-target 5 --device cpu  # smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from polyt5.chemistry import ScalableNoveltyIndex  # noqa: E402
from polyt5.evaluation.sweep import SweepPoint, run_sweep_point  # noqa: E402
from polyt5.inference import PolyT5PropertyPredictor  # noqa: E402
from polyt5.training import load_checkpoint  # noqa: E402
from polyt5.utils import RunDirectory, get_logger, load_config, select_device  # noqa: E402
from train_grpo import (  # noqa: E402
    build_reward_ensemble,
    build_verified_tokenizer,
    load_frozen_baseline,
    load_verified_model,
    verify_artifact,
)

#: The four Phase-3 arms this script can score, in the paper's C1-C4 order.
RLVR_ARMS: tuple[str, ...] = ("accuracy", "validity", "composite", "constraint")

#: Default novelty index (see ``scripts/train_grpo.py``'s copy of this constant
#: -- deliberately not imported from there, so a change to what an ARM trains
#: against does not silently change what this comparison screens against).
DEFAULT_NOVELTY_INDEX = REPO_ROOT / "artifacts" / "novelty" / "tg_generation_train"

#: TP window half-width. [PAPER] 50 K.
DEFAULT_TOLERANCE = 50.0

#: The metric each RLVR arm is compared against arm_b on, and which direction
#: is "better". [AMBIGUITY] the frozen baseline's success_criterion names "the
#: metric it optimized" without pinning one matrix column per arm, and Arm A/B
#: themselves only ever recorded three columns (property_mae, pv_rate,
#: tp_rate). The mapping below is this script's own, chosen against each arm's
#: actual reward definition in ``polyt5.rewards.composite``:
#:   accuracy   - continuous closeness to target Tg -> property_mae (lower).
#:   composite  - weighted sum of tg/pv/novelty, tg weighted 1.0 against pv's
#:                0.5 and novelty's 0.25 (DEFAULT_COMPOSITE_WEIGHTS) -> tg is
#:                the dominant term, so property_mae (lower) again.
#:   validity   - reward IS "cleared the full SV->TSD->DD->PV cascade" ->
#:                pv_rate (higher) is exactly that quantity.
#:   constraint - reward is an in-window AND synthesisable AND novel
#:                conjunction; "in-window" is precisely what tp_rate measures
#:                -> tp_rate (higher).
ARM_METRIC: dict[str, tuple[str, str]] = {
    "accuracy": ("property_mae", "lower"),
    "composite": ("property_mae", "lower"),
    "validity": ("pv_rate", "higher"),
    "constraint": ("tp_rate", "higher"),
}

MATRIX_COLUMNS: tuple[str, ...] = (
    "arm", "kind", "checkpoint", "temperature", "top_p",
    "n_requested", "n_pv", "pv_rate", "sr_rate", "duplicate_rate", "mean_length",
    "property_mean_auditor", "property_mae_auditor", "tp_rate_auditor",
    "property_mean_ensemble", "property_mae_ensemble", "tp_rate_ensemble",
    "optimized_metric", "optimized_value_auditor", "optimized_value_ensemble",
    "beats_arm_b", "survives_audit", "success",
)


def _resolve(path: str | Path) -> Path:
    """Resolve ``path`` relative to the repo root unless it is already absolute."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _latest_checkpoint(run_dir: Path) -> Path | None:
    """Return the highest-step checkpoint under a run directory, or ``None``."""
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return None
    candidates = sorted(checkpoints_dir.glob("step_*.pt"))
    return candidates[-1] if candidates else None


def load_rlvr_arm_model(arm_name: str, results_root: Path, logger):
    """Load an RLVR arm's trained policy, plus the temperature/top_p it trained with.

    Args:
        arm_name: One of :data:`RLVR_ARMS`.
        results_root: Root results directory (``results/`` by default); the
            arm's run directory is looked up as ``<results_root>/grpo_<arm>``.
        logger: Where to report what was found or why an arm was skipped.

    Returns:
        ``(model, checkpoint_path, SweepPoint)``, or ``None`` if the arm has no
        run directory or no checkpoint yet -- not an error, just "not trained
        yet"; the caller skips this arm and continues with the rest.
    """
    from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration

    config_path = REPO_ROOT / "configs" / "rl" / f"{arm_name}.yaml"
    cfg = load_config(config_path)
    experiment_name = cfg.get("experiment_name") or f"grpo_{arm_name}"
    run_dir = _resolve(results_root) / experiment_name

    checkpoint_path = _latest_checkpoint(run_dir)
    if checkpoint_path is None:
        logger.warning("arm %s: no checkpoint under %s -- skipping (not trained yet)",
                       arm_name, run_dir / "checkpoints")
        return None

    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    model_config = PolyT5Config.from_dict(payload["model_config"])
    model = PolyT5ForConditionalGeneration(model_config)
    model.load_state_dict(payload["model_state"])

    train_cfg = cfg.get("train", {})
    point = SweepPoint(
        temperature=float(train_cfg.get("temperature", 0.7)),
        top_p=float(train_cfg.get("top_p", 0.95)),
    )
    logger.info("arm %s: %s (step %s) T=%.2f top_p=%.2f", arm_name, checkpoint_path,
               payload.get("global_step"), point.temperature, point.top_p)
    return model, checkpoint_path, point


def evaluate_arm(
    model, tokenizer, *, arm_key: str, label: str, kind: str, point: SweepPoint,
    targets: list[float], n_samples: int, training_index, auditor, ensemble,
    device, batch_size: int, max_length: int, seed: int, checkpoint_label: str | None,
) -> dict[str, Any]:
    """Sample, screen and score one arm under the fixed protocol, twice.

    Args:
        model: The generation model for this arm (frozen ``generation``
            checkpoint for Arm A/B, the arm's own trained policy for the
            RLVR arms).
        tokenizer: The (verified) shared tokenizer.
        arm_key: One of ``"arm_a"``, ``"arm_b"``, or :data:`RLVR_ARMS`.
        label: Human-readable row label.
        kind: ``"baseline"`` or ``"rlvr"``.
        point: Sampling temperature/top_p for this arm.
        targets: Conditioning targets, e.g. ``[300.0, 400.0, 500.0]``.
        n_samples: Total candidates to generate (spread round-robin over
            ``targets``).
        training_index: TSD reference index, or ``None``.
        auditor: The held-out confirmation predictor.
        ensemble: The reward-ensemble predictor.
        device: Torch device.
        batch_size: Decoding batch size.
        max_length: Maximum generated tokens.
        seed: RNG seed; the same seed is used for both scoring passes so they
            decode IDENTICAL candidates (only the predictor differs).
        checkpoint_label: Checkpoint path recorded in the row, for provenance.

    Returns:
        One flat row dict, matching :data:`MATRIX_COLUMNS` (minus the
        ``beats_arm_b``/``survives_audit``/``success`` verdict columns, which
        the caller fills in once arm_b's row is known).
    """
    auditor_result = run_sweep_point(
        model, tokenizer, point=point, targets=targets, n_samples=n_samples,
        training_index=training_index, property_predictor=auditor, target_property=None,
        tolerance=DEFAULT_TOLERANCE, device=device, batch_size=batch_size, seed=seed,
        max_length=max_length,
    )
    ensemble_result = run_sweep_point(
        model, tokenizer, point=point, targets=targets, n_samples=n_samples,
        training_index=training_index, property_predictor=ensemble, target_property=None,
        tolerance=DEFAULT_TOLERANCE, device=device, batch_size=batch_size, seed=seed,
        max_length=max_length,
    )

    metric_name, _direction = ARM_METRIC.get(arm_key, (None, None))

    def _optimized(result, metric_name: str | None) -> float | None:
        if metric_name is None:
            return None
        if metric_name == "pv_rate":
            return result.counts.get("pv_rate")
        return getattr(result, metric_name)

    return {
        "arm": arm_key,
        "label": label,
        "kind": kind,
        "checkpoint": checkpoint_label,
        "temperature": point.temperature,
        "top_p": point.top_p,
        "n_requested": auditor_result.n_requested,
        "n_pv": auditor_result.counts.get("n_pv"),
        "pv_rate": auditor_result.counts.get("pv_rate"),
        "sr_rate": auditor_result.sr_rate,
        "duplicate_rate": auditor_result.duplicate_rate,
        "mean_length": auditor_result.mean_length,
        "property_mean_auditor": auditor_result.property_mean,
        "property_mae_auditor": auditor_result.property_mae,
        "tp_rate_auditor": auditor_result.tp_rate,
        "property_mean_ensemble": ensemble_result.property_mean,
        "property_mae_ensemble": ensemble_result.property_mae,
        "tp_rate_ensemble": ensemble_result.tp_rate,
        "optimized_metric": metric_name,
        "optimized_value_auditor": _optimized(auditor_result, metric_name),
        "optimized_value_ensemble": _optimized(ensemble_result, metric_name),
        "seconds": auditor_result.seconds + ensemble_result.seconds,
    }


def _metric_column(metric_name: str, predictor: str) -> str:
    """Map ``(metric_name, predictor)`` to the row column holding that value.

    ``pv_rate`` is purely structural (no predictor is involved in computing
    it), so both predictors share the single ``pv_rate`` column; every other
    metric has a separate ``<metric>_auditor`` / ``<metric>_ensemble`` column.
    """
    if metric_name == "pv_rate":
        return "pv_rate"
    return f"{metric_name}_{predictor}"


def apply_success_criterion(rows: list[dict[str, Any]]) -> None:
    """Fill in ``beats_arm_b`` / ``survives_audit`` / ``success`` in place.

    Reads arm_b's comparison value directly from the column its own row
    already has for the metric in question (e.g. ``property_mae_auditor``),
    NOT from arm_b's ``optimized_value_*`` -- arm_b has no entry in
    :data:`ARM_METRIC` (it optimizes nothing; it is supervised), so that
    column is always ``None`` on arm_b's own row.

    Args:
        rows: Rows produced by :func:`evaluate_arm`, arm_b's included.
            RLVR-arm rows are updated in place; Arm A/B rows get ``None`` for
            all three columns since the success criterion is not defined for
            them.
    """
    arm_b = next((row for row in rows if row["arm"] == "arm_b"), None)
    for row in rows:
        metric_name, direction = ARM_METRIC.get(row["arm"], (None, None))
        if metric_name is None or arm_b is None:
            row["beats_arm_b"] = None
            row["survives_audit"] = None
            row["success"] = None
            continue

        better = (lambda a, b: a < b) if direction == "lower" else (lambda a, b: a > b)
        own_ensemble = row.get(_metric_column(metric_name, "ensemble"))
        arm_b_ensemble = arm_b.get(_metric_column(metric_name, "ensemble"))
        own_auditor = row.get(_metric_column(metric_name, "auditor"))
        arm_b_auditor = arm_b.get(_metric_column(metric_name, "auditor"))

        beats_arm_b = (better(own_ensemble, arm_b_ensemble)
                       if own_ensemble is not None and arm_b_ensemble is not None else None)
        survives_audit = (better(own_auditor, arm_b_auditor)
                          if own_auditor is not None and arm_b_auditor is not None else None)
        row["beats_arm_b"] = beats_arm_b
        row["survives_audit"] = survives_audit
        row["success"] = (
            bool(beats_arm_b and survives_audit)
            if beats_arm_b is not None and survives_audit is not None else None
        )


def write_matrix_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write the arm x metric matrix as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MATRIX_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in MATRIX_COLUMNS})
    return path


def _format_cell(value: Any) -> str:
    """Render one markdown cell, never inventing a value for ``None``."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_matrix_markdown(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write the arm x metric matrix as a GitHub-flavoured markdown table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(MATRIX_COLUMNS) + " |",
        "|" + "|".join("---:" for _ in MATRIX_COLUMNS) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_format_cell(row.get(column)) for column in MATRIX_COLUMNS) + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arms", nargs="+", choices=RLVR_ARMS, default=list(RLVR_ARMS),
                        help="Which RLVR arms to attempt (baselines A/B are always included).")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results",
                        help="Root under which each arm's grpo_<arm> run directory lives.")
    parser.add_argument("--n-per-target", type=int, default=None,
                        help="Overrides evaluation_protocol.n_per_target (debug/smoke test).")
    parser.add_argument("--targets", type=float, nargs="+", default=None,
                        help="Overrides evaluation_protocol.targets_k (debug/smoke test).")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="[PAPER] TP window half-width, Kelvin.")
    parser.add_argument("--batch-size", type=int, default=64, help="Generation decode batch size.")
    parser.add_argument("--predictor-batch-size", type=int, default=64)
    parser.add_argument("--predictor-num-beams", type=int, default=4, help="[PAPER] 4.")
    parser.add_argument("--max-length", type=int, default=200, help="[PAPER] 200.")
    parser.add_argument("--novelty-index", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory. Defaults to results/arm_comparison.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = parse_args(argv)

    frozen_path = REPO_ROOT / "artifacts" / "baseline" / "frozen_baseline.json"
    try:
        frozen = load_frozen_baseline(frozen_path)
        tokenizer = build_verified_tokenizer(frozen)
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    device = select_device(args.device)
    out_root = args.out.parent if args.out else REPO_ROOT / "results"
    out_name = args.out.name if args.out else "arm_comparison"
    run_dir = RunDirectory.create(out_root, out_name)
    logger = get_logger("polyt5.compare_arms", log_file=run_dir.logs / "compare_arms.log")
    logger.info("device=%s out=%s", device, run_dir.root)

    protocol = frozen["evaluation_protocol"]
    targets = args.targets or [float(t) for t in protocol["targets_k"]]
    n_per_target = (args.n_per_target if args.n_per_target is not None
                    else int(protocol["n_per_target"]))
    n_samples = n_per_target * len(targets)
    logger.info("protocol: targets=%s n_per_target=%d -> n_samples=%d", targets, n_per_target,
               n_samples)

    tokenizer_path = _resolve(frozen["artifacts"]["tokenizer"]["path"])
    try:
        auditor_key = frozen["auditor"]
        auditor_meta = frozen["artifacts"][auditor_key]
        auditor_path = _resolve(auditor_meta["path"])
        verify_artifact(auditor_path, auditor_meta["sha256"], label=auditor_key)
        auditor = PolyT5PropertyPredictor.from_checkpoint(
            auditor_path, tokenizer_path, device=str(device),
            batch_size=args.predictor_batch_size, num_beams=args.predictor_num_beams,
            property_name="Tg",
        )
        ensemble = build_reward_ensemble(
            frozen, tokenizer_path, device=str(device), batch_size=args.predictor_batch_size,
            num_beams=args.predictor_num_beams,
        )
        generation_model, _ = load_verified_model(frozen, "generation")
    except (FileNotFoundError, ValueError, KeyError) as error:
        logger.error("could not build the comparison: %s", error)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    logger.info("auditor=%s (held out of every reward path) ensemble=%s", auditor_key,
               frozen["reward_ensemble"])

    novelty_path = args.novelty_index or DEFAULT_NOVELTY_INDEX
    training_index = ScalableNoveltyIndex.open(novelty_path) if Path(
        str(novelty_path) + ".json"
    ).is_file() or novelty_path.is_file() else None
    if training_index is None:
        logger.warning("novelty index not found at %s -- TSD/novelty will be a no-op",
                       novelty_path)

    rows: list[dict[str, Any]] = []
    for arm_key, label, sampling_key in (
        ("arm_a", "Arm A (default sampling)", "arm_a_default_sampling"),
        ("arm_b", "Arm B (tuned sampling)", "arm_b_tuned_sampling"),
    ):
        sampling = frozen[sampling_key]
        point = SweepPoint(temperature=float(sampling["temperature"]),
                           top_p=float(sampling["top_p"]))
        logger.info("%s: T=%.2f top_p=%.2f", label, point.temperature, point.top_p)
        row = evaluate_arm(
            generation_model, tokenizer, arm_key=arm_key, label=label, kind="baseline",
            point=point, targets=targets, n_samples=n_samples, training_index=training_index,
            auditor=auditor, ensemble=ensemble, device=device, batch_size=args.batch_size,
            max_length=args.max_length, seed=args.seed,
            checkpoint_label=str(frozen["artifacts"]["generation"]["path"]),
        )
        rows.append(row)

    for arm_name in args.arms:
        loaded = load_rlvr_arm_model(arm_name, args.results_root, logger)
        if loaded is None:
            rows.append({
                "arm": arm_name, "label": f"Arm {arm_name}", "kind": "rlvr",
                "checkpoint": None, "temperature": None, "top_p": None,
                "n_requested": None, "n_pv": None, "pv_rate": None, "sr_rate": None,
                "duplicate_rate": None, "mean_length": None,
                "property_mean_auditor": None, "property_mae_auditor": None,
                "tp_rate_auditor": None, "property_mean_ensemble": None,
                "property_mae_ensemble": None, "tp_rate_ensemble": None,
                "optimized_metric": ARM_METRIC.get(arm_name, (None, None))[0],
                "optimized_value_auditor": None, "optimized_value_ensemble": None,
            })
            continue
        model, checkpoint_path, point = loaded
        row = evaluate_arm(
            model, tokenizer, arm_key=arm_name, label=f"Arm {arm_name}", kind="rlvr",
            point=point, targets=targets, n_samples=n_samples, training_index=training_index,
            auditor=auditor, ensemble=ensemble, device=device, batch_size=args.batch_size,
            max_length=args.max_length, seed=args.seed, checkpoint_label=str(checkpoint_path),
        )
        rows.append(row)

    apply_success_criterion(rows)

    csv_path = write_matrix_csv(rows, run_dir.root / "matrix.csv")
    md_path = write_matrix_markdown(rows, run_dir.root / "matrix.md")
    summary = {
        "frozen_baseline": str(frozen_path),
        "auditor": auditor_key,
        "reward_ensemble": list(frozen["reward_ensemble"]),
        "success_criterion": frozen["success_criterion"],
        "evaluation_protocol": {"targets_k": targets, "n_per_target": n_per_target,
                                "n_samples": n_samples, "tolerance": args.tolerance},
        "arm_metric": {arm: {"metric": metric, "direction": direction}
                      for arm, (metric, direction) in ARM_METRIC.items()},
        "rows": rows,
    }
    summary_path = run_dir.root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    run_dir.write_manifest({"stage": "compare_arms", "arms_requested": list(args.arms)})

    logger.info("wrote %s, %s, %s", csv_path, md_path, summary_path)
    print(f"wrote {csv_path}\nwrote {md_path}\nwrote {summary_path}")
    for row in rows:
        if row["kind"] == "rlvr":
            print(f"  {row['arm']:<12} success={row.get('success')} "
                  f"metric={row.get('optimized_metric')} "
                  f"auditor={row.get('optimized_value_auditor')} "
                  f"ensemble={row.get('optimized_value_ensemble')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
