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
  ``scripts/train_grpo.py``), resampled at the temperature/``top_p`` the RUN
  ITSELF was trained with (read from ``<run_dir>/config.yaml`` -- the fully
  resolved config ``train_grpo.py`` writes next to its checkpoints, which
  reflects any ``--set`` override actually used at training time). An arm
  with no run directory yet is skipped with a warning rather than aborting
  the whole comparison -- this script is meant to run again as each arm
  finishes training, not only once everything is done.

Every predictor-dependent column is measured TWICE:

* by **the auditor** (``frozen_baseline.json``'s ``auditor`` split,
  ``tg_predictor_split4``) -- held out of every reward path, so this number
  cannot be an artifact of the reward model an RLVR arm was optimised against.
* by **the reward ensemble** (the same four splits ``scripts/train_grpo.py``
  builds the reward from) -- this is literally "the metric it optimized" for
  the RLVR arms, and gives a same-scale baseline for Arm A/B too.

[DECISION] (Ruling C) The auditor is a SINGLE model, so it has no ensemble
disagreement to report; every auditor-side prediction is fed to the reward
arms as ``(mean, std=0.0, n=1)``, which drives ``TgRewardConfig``'s confidence
weight to exactly 1.0. Concretely: the ensemble-scored ``composite_score`` /
accuracy reward is the TRUE, confidence-weighted objective an arm actually
optimizes; the auditor-scored version of the same quantity is UNWEIGHTED
closeness. This is deliberate, not an inconsistency -- confidence weighting is
a training-time steering device that keeps the policy away from predictions
its own ensemble cannot back; the auditor is answering a different question
(is the Tg claim true at all), so scoring it without that weighting is the
intended, and slightly STRICTER, comparison: the safe direction for a
reward-hacking check. See ``_auditor_predictions``.

The pre-registered ``success_criterion`` -- "An RLVR arm succeeds only if it
beats arm_b on the metric it optimized AND the gain survives scoring by the
auditor" -- is applied per RLVR arm as a two-clause test:

    beats_arm_b    = ensemble-scored optimized metric beats arm_b's
                     ensemble-scored value of the SAME metric
    survives_audit = auditor-scored optimized metric beats arm_b's
                     auditor-scored value of the SAME metric
    success        = beats_arm_b AND survives_audit

[DECISION] (Ruling D) For an arm whose optimized metric is purely STRUCTURAL
(``pv_rate`` -- RDKit chemistry, no predictor in the loop at all), clause 2 is
recorded as the literal string ``"N/A - structural metric, no predictor
involved"`` rather than a computed True/False: scoring it against the same
column clause 1 already used would manufacture a pass out of a tautology (both
sides ARE the same measurement by construction), which is worse than admitting
the check does not apply. ``success`` then reduces to ``beats_arm_b`` alone.
See ``STRUCTURAL_METRICS`` and :func:`apply_success_criterion`.

See ``ARM_METRIC`` for which column each arm is judged on and why; ``composite``
and ``constraint`` need PER-CANDIDATE scoring (not an aggregate the paper's
sweep machinery already returns), so this script reuses the actual
``polyt5.rewards`` arm objects -- ``build_reward_arm("composite", ...)`` /
``build_reward_arm("constraint", ...)`` -- built ONCE from the CANONICAL
``configs/rl/{composite,constraint}.yaml`` and applied identically to every
row, never from an RLVR arm's own (possibly ``--set``-overridden) training
config. That is what "computed identically for every row" requires: the
formula must be the same across rows, independent of how any one arm was
actually trained.

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

from polyt5.chemistry import ScalableNoveltyIndex, index_paths  # noqa: E402
from polyt5.evaluation.sweep import SweepPoint, property_columns, screen_candidates  # noqa: E402
from polyt5.inference import PolyT5PropertyPredictor  # noqa: E402
from polyt5.training import load_checkpoint  # noqa: E402
from polyt5.utils import RunDirectory, get_logger, load_config, select_device  # noqa: E402
from train_grpo import (  # noqa: E402
    build_reward_arm,
    build_reward_ensemble,
    build_verified_tokenizer,
    load_frozen_baseline,
    load_verified_model,
    sha256_of_file,
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

#: Metrics with NO predictor involved in computing them at all -- pure RDKit
#: chemistry. Both predictor-scored columns for these are definitionally the
#: same value; see Ruling D in the module docstring and
#: :func:`apply_success_criterion`.
STRUCTURAL_METRICS: frozenset[str] = frozenset({"pv_rate"})

#: The metric each RLVR arm is compared against arm_b on, and which direction
#: is "better", matched against each arm's actual reward definition in
#: ``polyt5.rewards.composite``:
#:   accuracy   - AccuracyArm's reward IS Tg closeness -> property_mae (lower).
#:   validity   - ValidityArm's reward IS "cleared the full SV->TSD->DD->PV
#:                cascade" -> pv_rate (higher); STRUCTURAL (see above).
#:   composite  - CompositeArm's reward is w_tg*r_tg + w_pv*pv_pass +
#:                w_novelty*novel -- a three-term objective with no existing
#:                aggregate column, so it is scored directly with the arm
#:                object itself -> composite_score (higher).
#:   constraint - ConstraintArm's reward is a conjunction over (|Tg-target| <=
#:                tolerance) AND (SA <= sa_max) AND novel -- again no existing
#:                aggregate captures the joint, so it is scored directly ->
#:                constraint_satisfaction_rate (higher).
ARM_METRIC: dict[str, tuple[str, str]] = {
    "accuracy": ("property_mae", "lower"),
    "validity": ("pv_rate", "higher"),
    "composite": ("composite_score", "higher"),
    "constraint": ("constraint_satisfaction_rate", "higher"),
}

MATRIX_COLUMNS: tuple[str, ...] = (
    "arm", "kind", "checkpoint", "checkpoint_sha256", "temperature", "top_p",
    "n_requested",
    "n_sv", "sv_rate", "n_tsd", "tsd_rate", "n_dd", "dd_rate", "n_pv", "pv_rate",
    "sr_rate", "duplicate_rate", "mean_length",
    "property_mean_auditor", "property_mae_auditor", "tp_rate_auditor",
    "property_mean_ensemble", "property_mae_ensemble", "tp_rate_ensemble",
    "composite_score_auditor", "composite_score_ensemble",
    "constraint_satisfaction_rate_auditor", "constraint_satisfaction_rate_ensemble",
    "novelty_index_sha256",
    "optimized_metric", "optimized_value_auditor", "optimized_value_ensemble",
    "beats_arm_b", "survives_audit", "success",
)


def _resolve(path: str | Path) -> Path:
    """Resolve ``path`` relative to the repo root unless it is already absolute."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _metric_column(metric_name: str, predictor: str) -> str:
    """Map ``(metric_name, predictor)`` to the row column holding that value.

    Structural metrics (see :data:`STRUCTURAL_METRICS`) are not computed by
    either predictor, so both share the single un-suffixed column; every
    other metric has a separate ``<metric>_auditor`` / ``<metric>_ensemble``
    column.
    """
    if metric_name in STRUCTURAL_METRICS:
        return metric_name
    return f"{metric_name}_{predictor}"


def _auditor_predictions(auditor, candidates: list[str]) -> list[tuple[float, float, int]]:
    """Build ``(mean, std, n)`` triples for a single-model predictor.

    [DECISION] (Ruling C) ``std`` is always ``0.0`` here -- the auditor is one
    model with nothing to disagree with itself about, so this is not a stand-in
    for a real uncertainty estimate. Feeding ``std=0.0`` into
    ``TgRewardConfig``'s confidence weight (``1 / (1 + std / sigma0)``) drives
    it to exactly ``1.0``, i.e. the composite/accuracy reward on the auditor
    side reduces to UNWEIGHTED closeness. See the module docstring for why
    that is the deliberate, and appropriately stricter, choice for clause 2 of
    the success criterion.

    Args:
        auditor: A single-model predictor (``Callable[[Sequence[str]],
            Sequence[float]]``, e.g. :class:`~polyt5.inference.
            PolyT5PropertyPredictor`).
        candidates: Raw generated PSELFIES strings.

    Returns:
        One ``(mean, 0.0, 1)`` triple per candidate, matching the
        ``EnsemblePropertyPredictor.predict_with_uncertainty`` contract that
        :mod:`polyt5.rewards` arms expect.
    """
    means = list(auditor(candidates))
    return [(float(mean), 0.0, 1) for mean in means]


def _score_with_arm(
    arm, candidates: list[str], sample_targets: list[float],
    predictions: list[tuple[float, float, int]],
) -> float:
    """Mean reward ``arm`` assigns across a FULL raw candidate batch.

    Mirrors exactly what :meth:`~polyt5.rl.trainer.GRPOTrainer.step` computes
    as its own ``reward_mean``: gated (structurally invalid) candidates
    contribute their gated reward -- normally ``0.0`` -- to the mean; they are
    not excluded from the denominator. That is "the metric it optimized", not
    a metric computed only over survivors.

    Args:
        arm: An :class:`~polyt5.rewards.ArmReward`, e.g.
            ``build_reward_arm("composite", ...)``.
        candidates: Raw generated PSELFIES strings, the FULL batch (not just
            PV survivors) -- ``CompositeArm``/``ConstraintArm`` do their own
            structural gating internally.
        sample_targets: Each candidate's own conditioning target, aligned
            with ``candidates``.
        predictions: ``(mean, std, n)`` per candidate, aligned with
            ``candidates``.

    Returns:
        The mean of ``arm(candidates, sample_targets, predictions)``'s
        ``.value``, or ``0.0`` for an empty batch.
    """
    if not candidates:
        return 0.0
    results = arm(candidates, sample_targets, predictions)
    return float(sum(result.value for result in results) / len(results))


def _latest_checkpoint(run_dir: Path) -> Path | None:
    """Return the highest-step checkpoint under a run directory, or ``None``."""
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return None
    candidates = sorted(checkpoints_dir.glob("step_*.pt"))
    return candidates[-1] if candidates else None


def load_rlvr_arm_model(arm_name: str, results_root: Path, tokenizer, logger):
    """Load an RLVR arm's trained policy, plus the temperature/top_p it actually trained with.

    Args:
        arm_name: One of :data:`RLVR_ARMS`.
        results_root: Root results directory (``results/`` by default); the
            arm's run directory is looked up as ``<results_root>/grpo_<arm>``.
        tokenizer: The tokenizer this comparison is scoring with -- the
            checkpoint's own recorded ``tokenizer_sha256`` is checked against
            it.
        logger: Where to report what was found or why an arm was skipped.

    Returns:
        ``(model, checkpoint_path, checkpoint_sha256, SweepPoint)``, or
        ``None`` if the arm has no run directory or no checkpoint yet (not an
        error -- "not trained yet") or if its checkpoint's recorded tokenizer
        does not match ``tokenizer`` (a real data-integrity problem, but one
        that should skip this one arm rather than abort the whole
        multi-arm comparison). Either way the caller skips this arm and
        continues with the rest.
    """
    from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration

    repo_config_path = REPO_ROOT / "configs" / "rl" / f"{arm_name}.yaml"
    repo_cfg = load_config(repo_config_path)
    experiment_name = repo_cfg.get("experiment_name") or f"grpo_{arm_name}"
    run_dir = _resolve(results_root) / experiment_name

    checkpoint_path = _latest_checkpoint(run_dir)
    if checkpoint_path is None:
        logger.warning("arm %s: no checkpoint under %s -- skipping (not trained yet)",
                       arm_name, run_dir / "checkpoints")
        return None

    # The RUN's own resolved config -- not the repo default -- is what
    # actually produced this checkpoint, including any --set override made at
    # training time (e.g. --set train.temperature=1.0). Falling back to the
    # repo config only when the run wrote none.
    run_config_path = run_dir / "config.yaml"
    if run_config_path.is_file():
        run_cfg = load_config(run_config_path)
    else:
        logger.warning("arm %s: no %s -- falling back to the repo config for sampling params",
                       arm_name, run_config_path)
        run_cfg = repo_cfg

    payload = load_checkpoint(checkpoint_path, map_location="cpu")

    recorded_sha = payload.get("tokenizer_sha256")
    if recorded_sha is not None and recorded_sha != tokenizer.sha256:
        logger.error(
            "arm %s: TOKENIZER MISMATCH -- %s was trained with vocabulary %s but this "
            "comparison is scoring with %s. Skipping this arm rather than silently "
            "corrupting every token id.", arm_name, checkpoint_path, recorded_sha[:16],
            tokenizer.sha256[:16],
        )
        return None

    model_config = PolyT5Config.from_dict(payload["model_config"])
    model = PolyT5ForConditionalGeneration(model_config)
    model.load_state_dict(payload["model_state"])

    train_cfg = run_cfg.get("train", {})
    point = SweepPoint(
        temperature=float(train_cfg.get("temperature", 0.7)),
        top_p=float(train_cfg.get("top_p", 0.95)),
    )
    checkpoint_sha256 = sha256_of_file(checkpoint_path)
    logger.info("arm %s: %s (step %s) T=%.2f top_p=%.2f", arm_name, checkpoint_path,
               payload.get("global_step"), point.temperature, point.top_p)
    return model, checkpoint_path, checkpoint_sha256, point


def evaluate_arm(
    model, tokenizer, *, arm_key: str, label: str, kind: str, point: SweepPoint,
    targets: list[float], n_samples: int, training_index, auditor, ensemble,
    composite_arm, constraint_arm, device, batch_size: int, max_length: int, seed: int,
    tolerance: float, checkpoint_label: str | None, checkpoint_sha256: str | None,
    novelty_index_sha256: str | None,
) -> dict[str, Any]:
    """Sample, screen and score one arm under the fixed protocol.

    Generates candidates exactly ONCE (a single :func:`~polyt5.evaluation.
    sweep.screen_candidates` call) and calls each predictor exactly ONCE over
    the full batch; every downstream column -- the own-target property stats,
    ``composite_score``, ``constraint_satisfaction_rate`` -- is derived from
    those two predictor passes rather than re-invoking either predictor.

    Args:
        model: The generation model for this arm (frozen ``generation``
            checkpoint for Arm A/B, the arm's own trained policy for the
            RLVR arms).
        tokenizer: The (verified) shared tokenizer.
        arm_key: ``"arm_a"``, ``"arm_b"``, or one of :data:`RLVR_ARMS`.
        label: Human-readable row label.
        kind: ``"baseline"`` or ``"rlvr"``.
        point: Sampling temperature/top_p for this arm.
        targets: Conditioning targets, e.g. ``[300.0, 400.0, 500.0]``.
        n_samples: Total candidates to generate (spread round-robin over
            ``targets``).
        training_index: TSD reference index, or ``None``.
        auditor: The held-out confirmation predictor (single model).
        ensemble: The reward-ensemble predictor.
        composite_arm: The CANONICAL ``build_reward_arm("composite", ...)``,
            shared across every row.
        constraint_arm: The CANONICAL ``build_reward_arm("constraint", ...)``,
            shared across every row.
        device: Torch device.
        batch_size: Decoding batch size.
        max_length: Maximum generated tokens.
        seed: RNG seed for generation.
        tolerance: TP window half-width, Kelvin.
        checkpoint_label: Checkpoint path recorded in the row, for provenance.
        checkpoint_sha256: Checkpoint content hash recorded in the row.
        novelty_index_sha256: Novelty index content hash recorded in the row.

    Returns:
        One flat row dict, matching :data:`MATRIX_COLUMNS` minus the
        ``beats_arm_b``/``survives_audit``/``success`` verdict columns, which
        the caller fills in once every row (arm_b's included) exists.
    """
    batch = screen_candidates(
        model, tokenizer, point=point, targets=targets, n_samples=n_samples,
        training_index=training_index, device=device, batch_size=batch_size, seed=seed,
        max_length=max_length, compute_sa=False,
    )
    candidates = batch.candidates
    sample_targets = batch.sample_targets

    auditor_means = list(auditor(candidates))
    auditor_predictions = [(float(mean), 0.0, 1) for mean in auditor_means]
    ensemble_predictions = list(ensemble.predict_with_uncertainty(candidates))

    screened_indices = [
        index for index, record in enumerate(batch.records)
        if record.passed_pv and record.canonical_psmiles is not None
    ]
    screened = batch.screened_psmiles
    screened_targets = batch.screened_targets
    auditor_screened = [auditor_means[index] for index in screened_indices]
    ensemble_screened = [ensemble_predictions[index][0] for index in screened_indices]

    # `property_columns` expects a Callable[[Sequence[str]], Sequence[float]];
    # these already-computed slices are handed back verbatim rather than
    # re-invoking either predictor on the (smaller) screened subset.
    tp_rate_auditor, property_mean_auditor, property_mae_auditor = property_columns(
        screened, screened_targets, lambda _candidates, values=auditor_screened: values,
        None, tolerance,
    )
    tp_rate_ensemble, property_mean_ensemble, property_mae_ensemble = property_columns(
        screened, screened_targets, lambda _candidates, values=ensemble_screened: values,
        None, tolerance,
    )

    composite_score_auditor = _score_with_arm(
        composite_arm, candidates, sample_targets, auditor_predictions
    )
    composite_score_ensemble = _score_with_arm(
        composite_arm, candidates, sample_targets, ensemble_predictions
    )
    constraint_satisfaction_rate_auditor = _score_with_arm(
        constraint_arm, candidates, sample_targets, auditor_predictions
    )
    constraint_satisfaction_rate_ensemble = _score_with_arm(
        constraint_arm, candidates, sample_targets, ensemble_predictions
    )

    row: dict[str, Any] = {
        "arm": arm_key,
        "label": label,
        "kind": kind,
        "checkpoint": checkpoint_label,
        "checkpoint_sha256": checkpoint_sha256,
        "temperature": point.temperature,
        "top_p": point.top_p,
        "n_requested": n_samples,
        **batch.counts.to_dict(),
        "sr_rate": batch.sr_rate,
        "duplicate_rate": batch.duplicate_rate,
        "mean_length": batch.mean_length,
        "property_mean_auditor": property_mean_auditor,
        "property_mae_auditor": property_mae_auditor,
        "tp_rate_auditor": tp_rate_auditor,
        "property_mean_ensemble": property_mean_ensemble,
        "property_mae_ensemble": property_mae_ensemble,
        "tp_rate_ensemble": tp_rate_ensemble,
        "composite_score_auditor": composite_score_auditor,
        "composite_score_ensemble": composite_score_ensemble,
        "constraint_satisfaction_rate_auditor": constraint_satisfaction_rate_auditor,
        "constraint_satisfaction_rate_ensemble": constraint_satisfaction_rate_ensemble,
        "novelty_index_sha256": novelty_index_sha256,
    }

    metric_name, _direction = ARM_METRIC.get(arm_key, (None, None))
    row["optimized_metric"] = metric_name
    row["optimized_value_auditor"] = (
        row.get(_metric_column(metric_name, "auditor")) if metric_name else None
    )
    row["optimized_value_ensemble"] = (
        row.get(_metric_column(metric_name, "ensemble")) if metric_name else None
    )
    return row


def apply_success_criterion(rows: list[dict[str, Any]]) -> None:
    """Fill in ``beats_arm_b`` / ``survives_audit`` / ``success`` in place.

    Reads arm_b's comparison value directly from the column its own row
    already has for the metric in question (e.g. ``property_mae_auditor``),
    NOT from arm_b's ``optimized_value_*`` -- arm_b has no entry in
    :data:`ARM_METRIC` (it optimizes nothing; it is supervised), so that
    column is always ``None`` on arm_b's own row.

    Structural metrics (:data:`STRUCTURAL_METRICS`) are a special case
    (Ruling D): ``survives_audit`` is recorded as the literal string
    ``"N/A - structural metric, no predictor involved"`` rather than a
    computed boolean, because both predictor columns for a structural metric
    ARE the same value by construction -- comparing them would manufacture an
    independently-confirmed pass out of a tautology. ``success`` then reduces
    to ``beats_arm_b`` alone for those arms.

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
        beats_arm_b = (better(own_ensemble, arm_b_ensemble)
                       if own_ensemble is not None and arm_b_ensemble is not None else None)

        if metric_name in STRUCTURAL_METRICS:
            row["beats_arm_b"] = beats_arm_b
            row["survives_audit"] = "N/A - structural metric, no predictor involved"
            row["success"] = beats_arm_b
            continue

        own_auditor = row.get(_metric_column(metric_name, "auditor"))
        arm_b_auditor = arm_b.get(_metric_column(metric_name, "auditor"))
        survives_audit = (better(own_auditor, arm_b_auditor)
                          if own_auditor is not None and arm_b_auditor is not None else None)
        row["beats_arm_b"] = beats_arm_b
        row["survives_audit"] = survives_audit
        row["success"] = (
            bool(beats_arm_b and survives_audit)
            if beats_arm_b is not None and survives_audit is not None else None
        )


def skipped_arm_row(arm_name: str) -> dict[str, Any]:
    """Build the placeholder row for an RLVR arm with no checkpoint yet.

    Derived from :data:`MATRIX_COLUMNS` (every column defaults to ``None``)
    rather than a hand-maintained key list, so a new column added there does
    not need a second place kept in sync.
    """
    row: dict[str, Any] = dict.fromkeys(MATRIX_COLUMNS)
    row.update({
        "arm": arm_name,
        "label": f"Arm {arm_name}",
        "kind": "rlvr",
        "optimized_metric": ARM_METRIC.get(arm_name, (None, None))[0],
    })
    return row


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
    parser.add_argument("--allow-missing-novelty-index", action="store_true",
                        help="Proceed even if the novelty index is missing, treating TSD as a "
                             "no-op for every row. Off by default: a missing index silently "
                             "inflates novelty across the entire matrix behind nothing but a "
                             "warning, so the run aborts unless this is explicitly passed.")
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
    tolerance = args.tolerance
    logger.info("protocol: targets=%s n_per_target=%d -> n_samples=%d tolerance=%.1f K", targets,
               n_per_target, n_samples, tolerance)

    # Ruling E: a missing novelty index silently turns TSD into a no-op for
    # EVERY row, inflating novelty across the whole matrix behind nothing but
    # a log line. Abort by default; --allow-missing-novelty-index opts in.
    # Checked HERE, before any checkpoint is loaded -- both to fail fast (no
    # point spending a minute loading five real checkpoints only to abort
    # anyway) and because this decision has nothing to do with them.
    novelty_path = args.novelty_index or DEFAULT_NOVELTY_INDEX
    try:
        training_index = ScalableNoveltyIndex.open(novelty_path)
        data_path, _meta_path = index_paths(novelty_path)
        novelty_index_sha256 = sha256_of_file(data_path)
        logger.info("novelty index: %s (sha256 %s...)", novelty_path, novelty_index_sha256[:16])
    except FileNotFoundError as error:
        if not args.allow_missing_novelty_index:
            print(
                f"ERROR: novelty index unusable: {error}\n"
                "A missing index silently turns TSD into a no-op for every row, inflating "
                "novelty across the entire matrix behind nothing but a warning. Build one "
                "with scripts/build_novelty_index.py, or pass --allow-missing-novelty-index "
                "to proceed anyway.",
                file=sys.stderr,
            )
            return 1
        logger.warning("novelty index unusable (%s) -- proceeding with TSD as a no-op for "
                       "every row (--allow-missing-novelty-index)", error)
        training_index = None
        novelty_index_sha256 = None

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

    # composite_score / constraint_satisfaction_rate must be computed
    # IDENTICALLY for every row -- see the module docstring -- so the two arm
    # objects are built ONCE here, from the canonical repo configs, and
    # reused for every row including arm_a and arm_b.
    composite_cfg = load_config(REPO_ROOT / "configs" / "rl" / "composite.yaml")
    constraint_cfg = load_config(REPO_ROOT / "configs" / "rl" / "constraint.yaml")
    composite_arm = build_reward_arm("composite", composite_cfg, novelty_index=training_index)
    constraint_arm = build_reward_arm("constraint", constraint_cfg, novelty_index=training_index)

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
            auditor=auditor, ensemble=ensemble, composite_arm=composite_arm,
            constraint_arm=constraint_arm, device=device, batch_size=args.batch_size,
            max_length=args.max_length, seed=args.seed, tolerance=tolerance,
            checkpoint_label=str(frozen["artifacts"]["generation"]["path"]),
            checkpoint_sha256=frozen["artifacts"]["generation"]["sha256"],
            novelty_index_sha256=novelty_index_sha256,
        )
        rows.append(row)

    for arm_name in args.arms:
        loaded = load_rlvr_arm_model(arm_name, args.results_root, tokenizer, logger)
        if loaded is None:
            rows.append(skipped_arm_row(arm_name))
            continue
        model, checkpoint_path, checkpoint_sha256, point = loaded
        row = evaluate_arm(
            model, tokenizer, arm_key=arm_name, label=f"Arm {arm_name}", kind="rlvr",
            point=point, targets=targets, n_samples=n_samples, training_index=training_index,
            auditor=auditor, ensemble=ensemble, composite_arm=composite_arm,
            constraint_arm=constraint_arm, device=device, batch_size=args.batch_size,
            max_length=args.max_length, seed=args.seed, tolerance=tolerance,
            checkpoint_label=str(checkpoint_path), checkpoint_sha256=checkpoint_sha256,
            novelty_index_sha256=novelty_index_sha256,
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
                                "n_samples": n_samples, "tolerance": tolerance},
        "novelty_index": {"path": str(novelty_path), "sha256": novelty_index_sha256,
                          "present": training_index is not None},
        "arm_metric": {arm: {"metric": metric, "direction": direction}
                      for arm, (metric, direction) in ARM_METRIC.items()},
        "structural_metrics": sorted(STRUCTURAL_METRICS),
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
