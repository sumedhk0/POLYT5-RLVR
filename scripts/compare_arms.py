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
  ``tg_predictor_split4``) -- held out of every reward PATH, so this number
  cannot be an artifact of the four particular models an RLVR arm was optimised
  against. It is not an independent confirmation of the Tg claim itself: the
  auditor is an independent random 80% sibling of the same corpus and shares
  ~80% of its training data with each reward model in expectation, so a hack
  exploiting a genuinely data-sparse region fools all five identically. See
  ``frozen_baseline.json``'s ``auditor_note``.
* by **the reward ensemble** (the same four splits ``scripts/train_grpo.py``
  builds the reward from) -- this is literally "the metric it optimized" for
  the RLVR arms, and gives a same-scale baseline for Arm A/B too.

[DECISION] (Ruling C) The auditor is a SINGLE model, so it has no ensemble
disagreement to report; every auditor-side prediction is fed to the reward
arms as ``(mean, std=0.0, n=1)``, and the auditor-side arm objects are built
with ``ensemble_size=1``. Coverage is then ``1/1`` and the spread is the
reported ``0.0``, which drives ``TgRewardConfig``'s confidence weight to
exactly 1.0. Concretely: the ensemble-scored ``accuracy_score`` /
``composite_score`` is the TRUE, confidence-weighted objective an arm actually
optimizes; the auditor-scored version of the same quantity is UNWEIGHTED
closeness. This is deliberate, not an inconsistency -- confidence weighting is
a training-time steering device that keeps the policy away from predictions
its own ensemble cannot back; the auditor is answering a different question
(is the Tg claim true at all), so scoring it without that weighting is the
intended, and slightly STRICTER, comparison: the safe direction for a
reward-hacking check. Note that "1 of 1" and "1 of 4" are deliberately NOT the
same case to :mod:`polyt5.rewards.tg` -- see its module docstring -- which is
exactly why the two sets of arm objects declare different ``ensemble_size``
values. See ``_auditor_predictions`` and :class:`ArmScorers`.

The pre-registered ``success_criterion`` -- "An RLVR arm succeeds only if it
beats arm_b on the metric it optimized AND the gain survives scoring by the
auditor" -- is applied per RLVR arm as a three-clause test:

    beats_arm_b    = ensemble-scored optimized metric beats arm_b's
                     ensemble-scored value of the SAME metric
    survives_audit = auditor-scored optimized metric beats arm_b's
                     auditor-scored value of the SAME metric
    success        = beats_arm_b AND survives_audit AND sampling_matches_arm_b

"Beats" is NOT a bare inequality. A bare ``<``/``>`` on point estimates from a
single generation seed records an 0.1 K win as ``success = True``,
indistinguishable from sampling noise -- arm_a and arm_b differ by 3.6 K and
MAE sampling error on ~840 survivors at RMSE ~44 K is order 1.5 K. Each clause
therefore requires BOTH an effect at least as large as the arm's
pre-registered ``min_margin`` AND a 95% percentile-bootstrap confidence
interval over candidates that excludes zero (:func:`_bootstrap_delta`). Both
the metric and the margin live in ``frozen_baseline.json``'s
``pre_registered_metrics``, not only in the mutable :data:`ARM_METRIC`;
:func:`check_pre_registration` asserts the two agree before anything is
measured, so a post-run edit to either one fails loudly instead of producing a
new, equally authoritative-looking ``summary.json``.

The third clause is finding 11: an RLVR arm is resampled at the temperature and
``top_p`` its own run trained with, and a single
``train_grpo.py --set train.temperature=1.0`` therefore produces a row that
cannot be compared to arm_b on ANY column (temperature alone moves MAE 7.1%
between arm_a and arm_b). ``sampling_matches_arm_b`` records that per row and
``success`` is refused -- left ``None``, not ``False`` -- when it is false: the
arm has not failed, it has not been measured comparably.

[DECISION] (Ruling D) For an arm whose optimized metric is purely STRUCTURAL
(``pv_rate`` -- RDKit chemistry, no predictor in the loop at all), clause 2 is
recorded as the literal string ``"N/A - structural metric, no predictor
involved"`` rather than a computed True/False: scoring it against the same
column clause 1 already used would manufacture a pass out of a tautology (both
sides ARE the same measurement by construction), which is worse than admitting
the check does not apply. ``success`` then reduces to ``beats_arm_b`` alone.
See ``STRUCTURAL_METRICS`` and :func:`apply_success_criterion`.

[DECISION] (Ruling D, constraint arm) ``constraint``'s optimized metric
(``constraint_satisfaction_rate``) is NOT structural -- it gets both an
``_auditor`` and an ``_ensemble`` column, unlike ``pv_rate`` above -- but it
is not a pure closeness score either, and its clause-2 comparison is
therefore a MIXED measurement, not a clean re-scoring of the same quantity by
a different predictor. :class:`~polyt5.rewards.composite.ConstraintArm`
evaluates a three-way conjunction (Tg window AND synthesisable AND novel) per
candidate: only the Tg clause reads from the injected ``predictions`` triple,
so ``constraint_satisfaction_rate_auditor`` uses the AUDITOR's mean for that
clause while its SA and novelty clauses are computed identically to the
ensemble-scored column -- straight from the candidate's own structure via
RDKit/the novelty index, not from either predictor. That is intentional: SA
and novelty have no predictor-side disagreement to audit (they are facts
about the SAME candidate regardless of which predictor scored the Tg clause),
so the only part of the conjunction clause 2 can independently confirm is Tg,
and it does. See :func:`_score_with_arm` and
:class:`~polyt5.rewards.composite.ConstraintArm`.

See ``ARM_METRIC`` for which column each arm is judged on and why. ``accuracy``,
``composite`` and ``constraint`` all need PER-CANDIDATE scoring (not an
aggregate the paper's sweep machinery already returns), so this script reuses
the actual ``polyt5.rewards`` arm objects -- ``build_reward_arm("accuracy",
...)`` / ``("composite", ...)`` / ``("constraint", ...)`` -- built ONCE from the
CANONICAL ``configs/rl/*.yaml`` and applied identically to every row, never
from an RLVR arm's own (possibly ``--set``-overridden) training config. That is
what "computed identically for every row" requires: the formula must be the
same across rows, independent of how any one arm was actually trained.

``accuracy_score`` was added for exactly the reason Task 8 added
``composite_score``: ``ARM_METRIC["accuracy"]`` used to be ``property_mae``,
which is not what C1 optimizes. C1's reward is ``mean(closeness x confidence)``
over the FULL rollout batch, with closeness clipped to zero beyond
``tolerance = 100 K`` and gated candidates contributing ``0.0``;
``property_mae`` is unweighted, unclipped and computed only over PV survivors,
so the two can move in opposite directions (measured: errors
``[10, 10, 120, 120] K`` beat errors ``[5, 5, 300, 300] K`` on MAE and lose on
C1's own reward). ``property_mae`` is KEPT as a reported column for every row --
it is the paper's headline conditioning metric and arm_b's frozen number is one
-- it is simply not C1's pre-registered success criterion.

Pre-registration integrity: hashed reward configs, not just a hashed record
-----------------------------------------------------------------------------
``frozen_baseline.json`` pins the metric/direction/min_margin per arm, but the
metric DEFINITIONS -- ``tolerance``, ``sigma0``, ``sigma_unknown``,
``min_coverage``, ``sa_max``, ``window_tolerance`` and (composite) ``weights``
-- live in ``configs/rl/accuracy.yaml`` / ``composite.yaml`` /
``constraint.yaml``, which are git-tracked but otherwise unfrozen. Editing one
silently changes what the pre-registered criterion MEANS, with no trace. This
script hashes exactly those three files (:func:`reward_config_paths`,
:func:`hash_reward_configs`) once, when it builds :class:`ArmScorers` from
them, and records the hash on every row (``*_reward_config_sha256``) and in
``summary.json`` (``reward_config_sha256``) -- so a run is auditable after the
fact for which reward definitions actually produced it. ``validity`` is not
included: its optimized metric (``pv_rate``) is structural RDKit chemistry
with no ``reward:`` block in the loop at all.

Drift columns for every row (adjudication (d))
-----------------------------------------------
``max_tanimoto_mean`` and ``near_copy_fraction`` used to exist only in the
training-time log (:mod:`polyt5.rl.drift`); arm_a and arm_b never ran under a
drift monitor, so the matrix had no memorisation signal for them at all. Since
novelty here is exact-canonical-match, a one-atom edit of a memorised polymer
scores full novelty -- these two columns are how a reader catches that. This
script builds one :class:`~polyt5.rl.drift.DriftMonitor` (reference
fingerprints only, no auditor) and calls it for EVERY row, arm_a's and arm_b's
included, reusing the same ``ensemble_predictions`` :func:`evaluate_arm`
already computed. They are diagnostic, not a success metric: neither is in
:data:`ARM_METRIC`, :data:`STRUCTURAL_METRICS`, or read by
:func:`apply_success_criterion`.

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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from polyt5.chemistry import ScalableNoveltyIndex, index_paths  # noqa: E402
from polyt5.evaluation.sweep import SweepPoint, property_columns, screen_candidates  # noqa: E402
from polyt5.inference import PolyT5PropertyPredictor  # noqa: E402
from polyt5.rl import DriftMonitor  # noqa: E402
from polyt5.rl.drift import DEFAULT_MAX_REFERENCE  # noqa: E402
from polyt5.training import load_checkpoint  # noqa: E402
from polyt5.utils import RunDirectory, get_logger, load_config, select_device  # noqa: E402
from train_grpo import (  # noqa: E402
    DEFAULT_DRIFT_REFERENCE,
    build_reward_arm,
    build_reward_ensemble,
    build_verified_tokenizer,
    load_drift_reference,
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
#:   accuracy   - AccuracyArm's reward IS mean(closeness x confidence) over the
#:                FULL batch, so it is scored directly with the arm object ->
#:                accuracy_score (higher). NOT property_mae; see the module
#:                docstring.
#:   validity   - ValidityArm's reward IS "cleared the full SV->TSD->DD->PV
#:                cascade" -> pv_rate (higher); STRUCTURAL (see above).
#:   composite  - CompositeArm's reward is w_tg*r_tg + w_pv*pv_pass +
#:                w_novelty*novel -- a three-term objective with no existing
#:                aggregate column, so it is scored directly with the arm
#:                object itself -> composite_score (higher).
#:   constraint - ConstraintArm's reward is a conjunction over (|Tg-target| <=
#:                tolerance) AND (SA <= sa_max) AND novel AND ensemble-backed --
#:                again no existing aggregate captures the joint, so it is
#:                scored directly -> constraint_satisfaction_rate (higher).
#:
#: This mapping is ordinary mutable Python and therefore is NOT the
#: pre-registration. ``frozen_baseline.json``'s ``pre_registered_metrics`` is;
#: :func:`check_pre_registration` refuses to run if the two disagree.
ARM_METRIC: dict[str, tuple[str, str]] = {
    "accuracy": ("accuracy_score", "higher"),
    "validity": ("pv_rate", "higher"),
    "composite": ("composite_score", "higher"),
    "constraint": ("constraint_satisfaction_rate", "higher"),
}

#: Minimum improvement over arm_b, in each metric's own units, below which a
#: difference is not called a win no matter how tight the bootstrap interval
#: is. Pre-registered in ``frozen_baseline.json`` alongside the metric names.
ARM_MIN_MARGIN: dict[str, float] = {
    "accuracy": 0.01,
    "validity": 0.02,
    "composite": 0.02,
    "constraint": 0.02,
}

#: Bootstrap settings, pre-registered in ``frozen_baseline.json``'s
#: ``success_criterion_statistics``.
N_BOOTSTRAP = 2000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 0

#: ``max_tanimoto_mean`` / ``near_copy_fraction`` (adjudication (d)) and the
#: three ``*_reward_config_sha256`` columns (pre-registration integrity, see
#: the module docstring) are DIAGNOSTIC / provenance columns computed
#: identically for every row, arm_a's and arm_b's included -- they are never
#: read by :data:`ARM_METRIC` or :func:`apply_success_criterion`.
MATRIX_COLUMNS: tuple[str, ...] = (
    "arm", "kind", "checkpoint", "checkpoint_sha256", "temperature", "top_p",
    "sampling_matches_arm_b", "n_requested",
    "n_sv", "sv_rate", "n_tsd", "tsd_rate", "n_dd", "dd_rate", "n_pv", "pv_rate",
    "sr_rate", "duplicate_rate", "max_tanimoto_mean", "near_copy_fraction", "mean_length",
    "property_mean_auditor", "property_mae_auditor", "tp_rate_auditor",
    "property_mean_ensemble", "property_mae_ensemble", "tp_rate_ensemble",
    "accuracy_score_auditor", "accuracy_score_ensemble",
    "composite_score_auditor", "composite_score_ensemble",
    "constraint_satisfaction_rate_auditor", "constraint_satisfaction_rate_ensemble",
    "novelty_index_sha256",
    "accuracy_reward_config_sha256", "composite_reward_config_sha256",
    "constraint_reward_config_sha256",
    "optimized_metric", "optimized_min_margin",
    "optimized_value_auditor", "optimized_value_ensemble",
    "delta_ensemble", "delta_ensemble_ci_low", "delta_ensemble_ci_high",
    "delta_auditor", "delta_auditor_ci_low", "delta_auditor_ci_high",
    "beats_arm_b", "survives_audit", "success",
)

#: Row key holding the per-candidate values every metric's mean is taken over,
#: which the bootstrap resamples. Kept off :data:`MATRIX_COLUMNS` (it is a
#: 1500-element list per metric per predictor, not a cell) and stripped before
#: ``summary.json`` is written.
SAMPLES_KEY = "_metric_samples"


@dataclass(frozen=True)
class ArmScorers:
    """The canonical reward-arm objects one predictor's triples are scored with.

    Two instances exist per run -- one for the auditor (built with
    ``ensemble_size=1``) and one for the reward ensemble (built with
    ``ensemble_size=len(ensemble)``) -- and each is shared by EVERY row,
    including arm_a's and arm_b's. That is what "computed identically for every
    row" means here: the formula is fixed across rows, while the ensemble size
    correctly describes the predictor whose triples are being scored. Feeding
    a four-member ensemble's triples to an arm declaring ``ensemble_size=1``
    raises in :func:`~polyt5.rewards.tg.tg_reward` rather than silently
    treating a one-member answer as full coverage.
    """

    accuracy: Any
    composite: Any
    constraint: Any


def _resolve(path: str | Path) -> Path:
    """Resolve ``path`` relative to the repo root unless it is already absolute."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


#: The three arms whose reward IS scored by an :class:`ArmScorers` object here.
#: ``validity`` is deliberately excluded: its optimized metric (``pv_rate``) is
#: structural RDKit chemistry with no ``reward:`` block involved at all -- see
#: :data:`STRUCTURAL_METRICS`.
REWARD_CONFIG_ARMS: tuple[str, ...] = ("accuracy", "composite", "constraint")


def reward_config_paths(repo_root: Path = REPO_ROOT) -> dict[str, Path]:
    """Canonical reward-definition YAMLs :func:`evaluate_arm`'s scorers are built from.

    [DECISION] These three files' ``tolerance``, ``sigma0``, ``sigma_unknown``,
    ``min_coverage``, ``sa_max``, ``window_tolerance`` and (composite only)
    ``weights`` are exactly what ``accuracy_score`` / ``composite_score`` /
    ``constraint_satisfaction_rate`` MEAN -- the pre-registered success metric
    for three of the four RLVR arms. Unlike ``frozen_baseline.json``, they are
    unhashed, unfrozen, git-tracked YAML: editing one silently changes what
    the pre-registered criterion measures, with no trace. Hashing them
    (:func:`hash_reward_configs`) and recording the hash in ``summary.json``
    and the matrix makes a run auditable after the fact for which reward
    definitions actually produced it.
    """
    return {name: repo_root / "configs" / "rl" / f"{name}.yaml" for name in REWARD_CONFIG_ARMS}


def hash_reward_configs(paths: dict[str, Path]) -> dict[str, str]:
    """SHA-256 of each canonical reward config in ``paths``, keyed by arm name."""
    return {name: sha256_of_file(path) for name, path in paths.items()}


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


def _auditor_predictions(means: Sequence[float]) -> list[tuple[float, float, int]]:
    """Build ``(mean, std, n)`` triples from a single-model predictor's raw means.

    [DECISION] (Ruling C) ``std`` is always ``0.0`` here -- the auditor is one
    model with nothing to disagree with itself about, so this is not a stand-in
    for a real uncertainty estimate. Feeding ``std=0.0`` into
    ``TgRewardConfig``'s confidence weight (``1 / (1 + std / sigma0)``) drives
    it to exactly ``1.0``, i.e. the composite/accuracy reward on the auditor
    side reduces to UNWEIGHTED closeness. See the module docstring for why
    that is the deliberate, and appropriately stricter, choice for clause 2 of
    the success criterion.

    Takes already-computed means rather than the auditor callable itself, so
    :func:`evaluate_arm` -- the only real caller -- can build these from the
    SAME ``auditor(candidates)`` call it also slices for ``auditor_screened``,
    instead of invoking the auditor a second time.

    Args:
        means: Raw auditor predictions, one per candidate.

    Returns:
        One ``(mean, 0.0, 1)`` triple per input, matching the
        ``EnsemblePropertyPredictor.predict_with_uncertainty`` contract that
        :mod:`polyt5.rewards` arms expect.
    """
    return [(float(mean), 0.0, 1) for mean in means]


def _arm_values(
    arm, candidates: list[str], sample_targets: list[float],
    predictions: list[tuple[float, float, int]],
) -> list[float]:
    """Per-candidate reward ``arm`` assigns across a FULL raw candidate batch.

    Mirrors exactly what :meth:`~polyt5.rl.trainer.GRPOTrainer.step` scores:
    gated (structurally invalid) candidates contribute their gated reward --
    normally ``0.0`` -- and are not excluded. That is "the metric it
    optimized", not a metric computed only over survivors.

    Args:
        arm: An :class:`~polyt5.rewards.ArmReward`, e.g.
            ``build_reward_arm("composite", ...)``.
        candidates: Raw generated PSELFIES strings, the FULL batch (not just
            PV survivors) -- the arms do their own structural gating.
        sample_targets: Each candidate's own conditioning target, aligned
            with ``candidates``.
        predictions: ``(mean, std, n)`` per candidate, aligned with
            ``candidates``.

    Returns:
        One float per candidate, in input order. Kept per-candidate rather than
        pre-averaged because the success criterion bootstraps over exactly
        these values (:func:`_bootstrap_delta`).
    """
    if not candidates:
        return []
    return [float(result.value) for result in arm(candidates, sample_targets, predictions)]


def _mean(values: Sequence[float]) -> float:
    """Mean of ``values``, or ``0.0`` for an empty sequence."""
    return float(sum(values) / len(values)) if len(values) else 0.0


def _abs_errors(predicted: Sequence[float], targets: Sequence[float]) -> list[float]:
    """Per-survivor ``|predicted - target|``, dropping non-finite predictions.

    Matches :func:`~polyt5.evaluation.sweep.property_columns`'s own accounting
    (finite pairs only), so ``mean(_abs_errors(...)) == property_mae``.
    """
    out: list[float] = []
    for value, target in zip(predicted, targets, strict=False):
        value = float(value)
        if value == value and value not in (float("inf"), float("-inf")):
            out.append(abs(value - float(target)))
    return out


def _score_with_arm(
    arm, candidates: list[str], sample_targets: list[float],
    predictions: list[tuple[float, float, int]],
) -> float:
    """Mean of :func:`_arm_values` -- the aggregate that lands in the matrix."""
    return _mean(_arm_values(arm, candidates, sample_targets, predictions))


def _bootstrap_delta(
    own: Sequence[float], base: Sequence[float], *, direction: str,
    n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> tuple[float | None, float | None, float | None]:
    """Signed improvement over ``base``, with a percentile bootstrap interval.

    Args:
        own: The arm's per-candidate metric values.
        base: arm_b's per-candidate values of the SAME metric, measured under
            the same protocol.
        direction: ``"higher"`` or ``"lower"`` -- which way is better. The
            returned delta is signed so that POSITIVE always means "better than
            arm_b", whichever direction that is, so one comparison rule works
            for every metric.
        n_boot: Bootstrap resamples.
        seed: Seed, so a recorded verdict is reproducible.
        confidence: Interval mass, e.g. ``0.95``.

    Returns:
        ``(delta, ci_low, ci_high)``, or ``(None, None, None)`` when either
        sample is empty -- "not measured", never a fabricated zero.

    Note:
        The two sides are resampled INDEPENDENTLY, not paired: the arms
        generate different candidates, so there is nothing to pair. The
        interval therefore covers sampling variation in the candidates within
        one generation seed. It does NOT cover seed-to-seed variation of the
        training runs themselves, which one run per arm cannot estimate; that
        remains a stated limitation of the study rather than something this
        interval quietly claims to have handled.
    """
    if len(own) == 0 or len(base) == 0:
        return None, None, None
    sign = 1.0 if direction == "higher" else -1.0
    own_array = np.asarray(own, dtype=float)
    base_array = np.asarray(base, dtype=float)
    delta = sign * float(own_array.mean() - base_array.mean())

    rng = np.random.default_rng(seed)
    own_means = rng.choice(own_array, size=(n_boot, own_array.size), replace=True).mean(axis=1)
    base_means = rng.choice(base_array, size=(n_boot, base_array.size), replace=True).mean(axis=1)
    deltas = sign * (own_means - base_means)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(deltas, [alpha, 1.0 - alpha])
    return delta, float(low), float(high)


def check_pre_registration(frozen: dict[str, Any]) -> list[str]:
    """Compare :data:`ARM_METRIC` / :data:`ARM_MIN_MARGIN` against the frozen record.

    The executable success criterion used to live entirely in ``ARM_METRIC`` --
    ordinary, mutable Python that nothing bound to the pre-registration, so a
    post-run edit produced a new ``summary.json`` that looked equally
    authoritative. ``frozen_baseline.json``'s ``pre_registered_metrics`` is now
    the record of what was promised, and this refuses to let the code drift
    from it.

    Args:
        frozen: The parsed ``frozen_baseline.json``.

    Returns:
        A list of human-readable disagreements. Empty means they match.
    """
    problems: list[str] = []
    registered = frozen.get("pre_registered_metrics")
    if not isinstance(registered, dict):
        return ["frozen_baseline.json has no pre_registered_metrics block"]

    coded = set(ARM_METRIC)
    pinned = {key for key in registered if not key.startswith("_")}
    if coded != pinned:
        problems.append(f"arms differ: code has {sorted(coded)}, frozen record has "
                        f"{sorted(pinned)}")
    for arm in sorted(coded & pinned):
        metric, direction = ARM_METRIC[arm]
        entry = registered[arm]
        if entry.get("metric") != metric:
            problems.append(f"{arm}: code metric {metric!r} != frozen {entry.get('metric')!r}")
        if entry.get("direction") != direction:
            problems.append(
                f"{arm}: code direction {direction!r} != frozen {entry.get('direction')!r}")
        if float(entry.get("min_margin", -1.0)) != float(ARM_MIN_MARGIN.get(arm, -1.0)):
            problems.append(
                f"{arm}: code min_margin {ARM_MIN_MARGIN.get(arm)!r} != frozen "
                f"{entry.get('min_margin')!r}")

    frozen_structural = frozen.get("structural_metrics")
    if frozen_structural is not None and sorted(STRUCTURAL_METRICS) != sorted(frozen_structural):
        problems.append(f"structural_metrics differ: code {sorted(STRUCTURAL_METRICS)} != frozen "
                        f"{sorted(frozen_structural)}")

    statistics = frozen.get("success_criterion_statistics", {})
    for key, coded_value in (("n_bootstrap", N_BOOTSTRAP), ("confidence", BOOTSTRAP_CONFIDENCE),
                             ("bootstrap_seed", BOOTSTRAP_SEED)):
        if key in statistics and statistics[key] != coded_value:
            problems.append(f"{key}: code {coded_value!r} != frozen {statistics[key]!r}")
    return problems


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
    auditor_scorers: ArmScorers, ensemble_scorers: ArmScorers,
    similarity_monitor: DriftMonitor,
    device, batch_size: int, max_length: int, seed: int,
    tolerance: float, checkpoint_label: str | None, checkpoint_sha256: str | None,
    novelty_index_sha256: str | None, reward_config_sha256: dict[str, str],
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
        auditor_scorers: The CANONICAL arm objects for scoring the auditor's
            ``(mean, 0.0, 1)`` triples (``ensemble_size=1``), shared across
            every row.
        ensemble_scorers: The CANONICAL arm objects for scoring the reward
            ensemble's triples (``ensemble_size=len(ensemble)``), shared
            across every row.
        similarity_monitor: A :class:`~polyt5.rl.drift.DriftMonitor` built
            with no auditor (only the reference fingerprints), shared across
            every row -- see the module docstring's adjudication (d).
            Supplies ``max_tanimoto_mean`` / ``near_copy_fraction``, computed
            here for EVERY row, arm_a's and arm_b's included, since neither
            ran under a drift monitor during its own (Phase 1/2) evaluation.
        device: Torch device.
        batch_size: Decoding batch size.
        max_length: Maximum generated tokens.
        seed: RNG seed for generation.
        tolerance: TP window half-width, Kelvin.
        checkpoint_label: Checkpoint path recorded in the row, for provenance.
        checkpoint_sha256: Checkpoint content hash recorded in the row.
        novelty_index_sha256: Novelty index content hash recorded in the row.
        reward_config_sha256: ``{"accuracy": ..., "composite": ...,
            "constraint": ...}`` SHA-256 of the canonical
            ``configs/rl/*.yaml`` files ``auditor_scorers`` /
            ``ensemble_scorers`` were built from (:func:`hash_reward_configs`),
            recorded verbatim on every row for pre-registration provenance.

    Returns:
        One flat row dict, matching :data:`MATRIX_COLUMNS` minus the verdict
        columns (``sampling_matches_arm_b``, the ``delta_*`` pairs,
        ``beats_arm_b``/``survives_audit``/``success``), which
        :func:`apply_success_criterion` fills in once every row (arm_b's
        included) exists. The row also carries :data:`SAMPLES_KEY` -- the
        per-candidate values behind every metric's mean, which the bootstrap
        resamples; it is stripped before ``summary.json`` is written.
    """
    batch = screen_candidates(
        model, tokenizer, point=point, targets=targets, n_samples=n_samples,
        training_index=training_index, device=device, batch_size=batch_size, seed=seed,
        max_length=max_length, compute_sa=False,
    )
    candidates = batch.candidates
    sample_targets = batch.sample_targets

    auditor_means = list(auditor(candidates))
    auditor_predictions = _auditor_predictions(auditor_means)
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

    # Per-candidate values, kept rather than pre-averaged: every matrix cell
    # below is their mean, and the success criterion's bootstrap resamples
    # exactly these. The auditor's are computed before the ensemble's for both
    # arms (see test_evaluate_arm_auditor_predictions_are_wired_through_the_helper).
    accuracy_values_auditor = _arm_values(
        auditor_scorers.accuracy, candidates, sample_targets, auditor_predictions)
    accuracy_values_ensemble = _arm_values(
        ensemble_scorers.accuracy, candidates, sample_targets, ensemble_predictions)
    composite_values_auditor = _arm_values(
        auditor_scorers.composite, candidates, sample_targets, auditor_predictions)
    composite_values_ensemble = _arm_values(
        ensemble_scorers.composite, candidates, sample_targets, ensemble_predictions)
    constraint_values_auditor = _arm_values(
        auditor_scorers.constraint, candidates, sample_targets, auditor_predictions)
    constraint_values_ensemble = _arm_values(
        ensemble_scorers.constraint, candidates, sample_targets, ensemble_predictions)

    # pv_rate is structural: one 0/1 indicator per generated candidate, over
    # the full batch, whose mean IS batch.counts' pv_rate. Both predictor
    # slots share it, exactly as _metric_column already says.
    pv_indicators = [float(record.passed_pv) for record in batch.records]
    # property_mae's samples are per-SURVIVOR absolute errors -- the same
    # finite pairs property_columns averages -- which is precisely why it is
    # not any arm's success metric: its denominator moves with validity.
    mae_values_auditor = _abs_errors(auditor_screened, screened_targets)
    mae_values_ensemble = _abs_errors(ensemble_screened, screened_targets)

    composite_score_auditor = _mean(composite_values_auditor)
    composite_score_ensemble = _mean(composite_values_ensemble)
    constraint_satisfaction_rate_auditor = _mean(constraint_values_auditor)
    constraint_satisfaction_rate_ensemble = _mean(constraint_values_ensemble)

    # Adjudication (d): diagnostic only, computed identically for every row
    # (arm_a's and arm_b's included -- neither ran under a drift monitor
    # during its own Phase 1/2 evaluation) via the SAME module the training-
    # time drift monitor uses. `ensemble_predictions` is reused, not
    # recomputed -- the auditor-gap half is unused here (this monitor was
    # built with no auditor) and its keys come back `None`.
    drift_stats = similarity_monitor.observe(candidates, ensemble_predictions)

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
        "max_tanimoto_mean": drift_stats["max_tanimoto_mean"],
        "near_copy_fraction": drift_stats["near_copy_fraction"],
        "mean_length": batch.mean_length,
        "property_mean_auditor": property_mean_auditor,
        "property_mae_auditor": property_mae_auditor,
        "tp_rate_auditor": tp_rate_auditor,
        "property_mean_ensemble": property_mean_ensemble,
        "property_mae_ensemble": property_mae_ensemble,
        "tp_rate_ensemble": tp_rate_ensemble,
        "accuracy_score_auditor": _mean(accuracy_values_auditor),
        "accuracy_score_ensemble": _mean(accuracy_values_ensemble),
        "composite_score_auditor": composite_score_auditor,
        "composite_score_ensemble": composite_score_ensemble,
        "constraint_satisfaction_rate_auditor": constraint_satisfaction_rate_auditor,
        "constraint_satisfaction_rate_ensemble": constraint_satisfaction_rate_ensemble,
        "novelty_index_sha256": novelty_index_sha256,
        "accuracy_reward_config_sha256": reward_config_sha256.get("accuracy"),
        "composite_reward_config_sha256": reward_config_sha256.get("composite"),
        "constraint_reward_config_sha256": reward_config_sha256.get("constraint"),
        SAMPLES_KEY: {
            "accuracy_score": {"auditor": accuracy_values_auditor,
                               "ensemble": accuracy_values_ensemble},
            "composite_score": {"auditor": composite_values_auditor,
                                "ensemble": composite_values_ensemble},
            "constraint_satisfaction_rate": {"auditor": constraint_values_auditor,
                                             "ensemble": constraint_values_ensemble},
            "pv_rate": {"auditor": pv_indicators, "ensemble": pv_indicators},
            "property_mae": {"auditor": mae_values_auditor,
                             "ensemble": mae_values_ensemble},
        },
    }

    metric_name, _direction = ARM_METRIC.get(arm_key, (None, None))
    row["optimized_metric"] = metric_name
    row["optimized_min_margin"] = ARM_MIN_MARGIN.get(arm_key)
    row["optimized_value_auditor"] = (
        row.get(_metric_column(metric_name, "auditor")) if metric_name else None
    )
    row["optimized_value_ensemble"] = (
        row.get(_metric_column(metric_name, "ensemble")) if metric_name else None
    )
    return row


#: Recorded in ``survives_audit`` for an arm whose optimized metric is purely
#: structural -- see Ruling D in the module docstring.
STRUCTURAL_AUDIT_NOTE = "N/A - structural metric, no predictor involved"


def _samples_for(row: dict[str, Any], metric_name: str, predictor: str) -> list[float]:
    """The per-candidate values behind one metric cell, or ``[]`` if absent."""
    samples = row.get(SAMPLES_KEY) or {}
    return list((samples.get(metric_name) or {}).get(predictor) or [])


def _clause(
    row: dict[str, Any], arm_b: dict[str, Any], metric_name: str, direction: str,
    predictor: str, min_margin: float, *, n_boot: int, seed: int, confidence: float,
) -> tuple[bool | None, float | None, float | None, float | None]:
    """Decide one clause of the success criterion, and return its evidence.

    Args:
        row: The RLVR arm's row.
        arm_b: arm_b's row.
        metric_name: The pre-registered metric.
        direction: ``"higher"`` or ``"lower"``.
        predictor: ``"auditor"`` or ``"ensemble"``.
        min_margin: The pre-registered minimum effect size.
        n_boot: Bootstrap resamples.
        seed: Bootstrap seed.
        confidence: Interval mass.

    Returns:
        ``(verdict, delta, ci_low, ci_high)``. ``verdict`` is ``None`` -- not
        ``False`` -- when the clause could not be evaluated at all, because
        "not measured" and "measured and lost" are different facts.
    """
    own = _samples_for(row, metric_name, predictor)
    base = _samples_for(arm_b, metric_name, predictor)
    delta, low, high = _bootstrap_delta(
        own, base, direction=direction, n_boot=n_boot, seed=seed, confidence=confidence)
    if delta is None or low is None:
        return None, delta, low, high
    # BOTH conditions, deliberately: the CI alone would call a 0.001 win
    # significant at n=1500, and the margin alone would call a large but
    # entirely noise-driven difference a win.
    return bool(delta >= min_margin and low > 0.0), delta, low, high


def apply_success_criterion(
    rows: list[dict[str, Any]], *, n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> None:
    """Fill in the verdict columns in place, with their statistical evidence.

    Each clause requires BOTH an improvement over arm_b of at least the arm's
    pre-registered :data:`ARM_MIN_MARGIN` AND a bootstrap confidence interval
    on that improvement that excludes zero. A bare inequality on point
    estimates from one generation seed cannot separate an 0.1 K win from
    sampling noise, and the study's whole premise is a trustworthy comparison.

    arm_b's comparison values come from the per-candidate samples on its own
    row (and its aggregate from its own metric column), NOT from arm_b's
    ``optimized_value_*`` -- arm_b has no entry in :data:`ARM_METRIC` (it
    optimizes nothing; it is supervised), so that column is always ``None`` on
    its row.

    Structural metrics (:data:`STRUCTURAL_METRICS`) are a special case
    (Ruling D): ``survives_audit`` is recorded as
    :data:`STRUCTURAL_AUDIT_NOTE` rather than a computed boolean, because both
    predictor columns for a structural metric ARE the same value by
    construction -- comparing them would manufacture an independently-confirmed
    pass out of a tautology. ``success`` then reduces to ``beats_arm_b`` alone
    for those arms.

    ``sampling_matches_arm_b`` is the third clause (finding 11). An RLVR arm is
    resampled at whatever temperature/``top_p`` its own run trained with, so a
    ``--set train.temperature=1.0`` run yields a row that is not comparable to
    arm_b on any column. ``success`` is then left ``None``, not ``False``: the
    arm has not lost, it has not been measured against the same operating
    point.

    Args:
        rows: Rows produced by :func:`evaluate_arm`, arm_b's included.
            RLVR-arm rows are updated in place; Arm A/B rows get ``None`` for
            the verdict columns since the success criterion is not defined for
            them.
        n_boot: Bootstrap resamples.
        seed: Bootstrap seed, so a recorded verdict is reproducible.
        confidence: Interval mass.
    """
    arm_b = next((row for row in rows if row["arm"] == "arm_b"), None)
    for row in rows:
        row["sampling_matches_arm_b"] = _sampling_matches(row, arm_b)
        for column in ("delta_ensemble", "delta_ensemble_ci_low", "delta_ensemble_ci_high",
                       "delta_auditor", "delta_auditor_ci_low", "delta_auditor_ci_high"):
            row.setdefault(column, None)

        metric_name, direction = ARM_METRIC.get(row["arm"], (None, None))
        if metric_name is None or arm_b is None:
            row["beats_arm_b"] = None
            row["survives_audit"] = None
            row["success"] = None
            continue

        min_margin = float(ARM_MIN_MARGIN.get(row["arm"], 0.0))
        row["optimized_min_margin"] = min_margin
        beats_arm_b, delta, low, high = _clause(
            row, arm_b, metric_name, direction, "ensemble", min_margin,
            n_boot=n_boot, seed=seed, confidence=confidence)
        row["delta_ensemble"] = delta
        row["delta_ensemble_ci_low"] = low
        row["delta_ensemble_ci_high"] = high

        if metric_name in STRUCTURAL_METRICS:
            survives_audit: Any = STRUCTURAL_AUDIT_NOTE
            audit_clause: bool | None = True
        else:
            audit_clause, delta_a, low_a, high_a = _clause(
                row, arm_b, metric_name, direction, "auditor", min_margin,
                n_boot=n_boot, seed=seed, confidence=confidence)
            row["delta_auditor"] = delta_a
            row["delta_auditor_ci_low"] = low_a
            row["delta_auditor_ci_high"] = high_a
            survives_audit = audit_clause

        row["beats_arm_b"] = beats_arm_b
        row["survives_audit"] = survives_audit
        if beats_arm_b is None or audit_clause is None or row["sampling_matches_arm_b"] is None:
            row["success"] = None
        elif not row["sampling_matches_arm_b"]:
            # Refused, not failed: an incomparable sampling point says nothing
            # about whether the arm worked.
            row["success"] = None
        else:
            row["success"] = bool(beats_arm_b and audit_clause)


def _sampling_matches(row: dict[str, Any], arm_b: dict[str, Any] | None) -> bool | None:
    """Whether this row was sampled at arm_b's operating point.

    Returns ``None`` when either point is unknown (an untrained arm's
    placeholder row, or a missing arm_b), and ``True`` on arm_b's own row.
    """
    if arm_b is None:
        return None
    own = (row.get("temperature"), row.get("top_p"))
    base = (arm_b.get("temperature"), arm_b.get("top_p"))
    if None in own or None in base:
        return None
    return float(own[0]) == float(base[0]) and float(own[1]) == float(base[1])


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
        "optimized_min_margin": ARM_MIN_MARGIN.get(arm_name),
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
    parser.add_argument("--drift-reference", type=Path, default=None,
                        help="Labelled Tg polymers to compute max_tanimoto_mean/"
                             "near_copy_fraction against (diagnostic columns only -- see "
                             "STRUCTURAL_METRICS and the module docstring; never wired into "
                             "ARM_METRIC or the success criterion). Defaults to "
                             "train_grpo.DEFAULT_DRIFT_REFERENCE, the same file every arm's "
                             "drift: block points at. Missing/unusable degrades the two columns "
                             "to None for every row rather than aborting -- this is a diagnostic, "
                             "not a gate.")
    parser.add_argument("--max-drift-reference", type=int, default=DEFAULT_MAX_REFERENCE)
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

    # The success criterion is pre-registered in the frozen record, not in this
    # file. If the two have drifted apart, every verdict below would be
    # computed against a criterion nobody promised -- refuse before measuring.
    pre_registration_problems = check_pre_registration(frozen)
    if pre_registration_problems:
        print(
            "ERROR: ARM_METRIC / ARM_MIN_MARGIN disagree with the pre-registered criterion in "
            f"{frozen_path}:\n  " + "\n  ".join(pre_registration_problems)
            + "\nThe frozen record is the pre-registration. Changing the criterion after "
              "seeing results is post-hoc metric selection; fix the code, not the record.",
            file=sys.stderr,
        )
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

    # accuracy_score / composite_score / constraint_satisfaction_rate must be
    # computed IDENTICALLY for every row -- see the module docstring -- so the
    # arm objects are built ONCE here, from the canonical repo configs, and
    # reused for every row including arm_a and arm_b. Two sets, differing only
    # in the ensemble_size that describes the predictor whose triples they will
    # score: the auditor is one model (n=1 of 1 is full coverage), the reward
    # ensemble is four (n=1 of 4 means three members could not read the
    # candidate, which is the opposite situation -- see polyt5.rewards.tg).
    ensemble_size = len(ensemble)
    # Pre-registration integrity: these three YAMLs are what accuracy_score /
    # composite_score / constraint_satisfaction_rate MEAN, and they are
    # unhashed, unfrozen, git-tracked files -- see reward_config_paths's
    # docstring. Hashed here, ONCE, alongside loading them, so the hash on
    # every row is provably the hash of the file the scorers below were
    # actually built from.
    reward_config_paths_map = reward_config_paths()
    reward_config_sha256 = hash_reward_configs(reward_config_paths_map)
    arm_configs = {
        name: load_config(path) for name, path in reward_config_paths_map.items()
    }
    logger.info("reward config hashes: %s", reward_config_sha256)

    def _scorers(size: int) -> ArmScorers:
        return ArmScorers(**{
            name: build_reward_arm(name, arm_configs[name], novelty_index=training_index,
                                   ensemble_size=size)
            for name in ("accuracy", "composite", "constraint")
        })

    auditor_scorers = _scorers(1)
    ensemble_scorers = _scorers(ensemble_size)
    logger.info("reward-arm scorers: auditor ensemble_size=1, ensemble ensemble_size=%d",
                ensemble_size)

    # Adjudication (d): max_tanimoto_mean / near_copy_fraction, computed
    # identically for every row (arm_a's and arm_b's included) via the same
    # DriftMonitor the training-time monitor uses -- built with NO auditor
    # (only the reference fingerprints), so it costs zero extra predictor
    # calls; ensemble_predictions is reused from what evaluate_arm already
    # computes. A missing/unusable reference degrades the two columns to
    # None for every row rather than aborting -- this is a diagnostic, not a
    # pre-registered success metric (see STRUCTURAL_METRICS and ARM_METRIC).
    drift_reference_path = args.drift_reference or DEFAULT_DRIFT_REFERENCE
    drift_reference = load_drift_reference(
        _resolve(drift_reference_path), limit=args.max_drift_reference)
    similarity_monitor = DriftMonitor(
        reference_psmiles=drift_reference, max_reference=args.max_drift_reference,
        seed=args.seed,
    )
    if similarity_monitor.has_similarity:
        logger.info("drift reference: %s (%d fingerprints)", drift_reference_path,
                    similarity_monitor.n_reference)
    else:
        logger.warning(
            "drift reference unusable (%s) -- max_tanimoto_mean/near_copy_fraction will be "
            "None for every row (diagnostic only, does not block the run)", drift_reference_path)

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
            auditor=auditor, ensemble=ensemble, auditor_scorers=auditor_scorers,
            ensemble_scorers=ensemble_scorers, similarity_monitor=similarity_monitor,
            device=device, batch_size=args.batch_size,
            max_length=args.max_length, seed=args.seed, tolerance=tolerance,
            checkpoint_label=str(frozen["artifacts"]["generation"]["path"]),
            checkpoint_sha256=frozen["artifacts"]["generation"]["sha256"],
            novelty_index_sha256=novelty_index_sha256,
            reward_config_sha256=reward_config_sha256,
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
            auditor=auditor, ensemble=ensemble, auditor_scorers=auditor_scorers,
            ensemble_scorers=ensemble_scorers, similarity_monitor=similarity_monitor,
            device=device, batch_size=args.batch_size,
            max_length=args.max_length, seed=args.seed, tolerance=tolerance,
            checkpoint_label=str(checkpoint_path), checkpoint_sha256=checkpoint_sha256,
            novelty_index_sha256=novelty_index_sha256,
            reward_config_sha256=reward_config_sha256,
        )
        rows.append(row)

    apply_success_criterion(rows)

    for row in rows:
        if row.get("kind") == "rlvr" and row.get("sampling_matches_arm_b") is False:
            logger.error(
                "arm %s was sampled at T=%s top_p=%s but arm_b is T=%s top_p=%s -- this row is "
                "NOT comparable to arm_b on any column (temperature alone moves MAE 7.1%% "
                "between arm_a and arm_b). success is recorded as null, not false.",
                row["arm"], row.get("temperature"), row.get("top_p"),
                next((r.get("temperature") for r in rows if r["arm"] == "arm_b"), None),
                next((r.get("top_p") for r in rows if r["arm"] == "arm_b"), None),
            )

    csv_path = write_matrix_csv(rows, run_dir.root / "matrix.csv")
    md_path = write_matrix_markdown(rows, run_dir.root / "matrix.md")
    # The per-candidate sample vectors are ~1500 floats per metric per
    # predictor per row; they exist for the bootstrap, not for the record.
    summary_rows = [{k: v for k, v in row.items() if k != SAMPLES_KEY} for row in rows]
    summary = {
        "frozen_baseline": str(frozen_path),
        "auditor": auditor_key,
        "auditor_note": frozen.get("auditor_note"),
        "reward_ensemble": list(frozen["reward_ensemble"]),
        "reward_ensemble_size": ensemble_size,
        "success_criterion": frozen["success_criterion"],
        "success_criterion_statistics": {
            "test": "percentile bootstrap over candidates",
            "n_bootstrap": N_BOOTSTRAP,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "rule": "clause holds iff delta >= min_margin AND ci_low > 0; success also "
                    "requires sampling_matches_arm_b",
        },
        "evaluation_protocol": {"targets_k": targets, "n_per_target": n_per_target,
                                "n_samples": n_samples, "tolerance": tolerance},
        "novelty_index": {"path": str(novelty_path), "sha256": novelty_index_sha256,
                          "present": training_index is not None},
        "arm_metric": {arm: {"metric": metric, "direction": direction,
                             "min_margin": ARM_MIN_MARGIN.get(arm)}
                      for arm, (metric, direction) in ARM_METRIC.items()},
        "pre_registration_verified_against": str(frozen_path),
        "structural_metrics": sorted(STRUCTURAL_METRICS),
        # Pre-registration integrity (see reward_config_paths's docstring):
        # accuracy_score / composite_score / constraint_satisfaction_rate's
        # DEFINITIONS -- tolerance, sigma0, sigma_unknown, min_coverage,
        # sa_max, window_tolerance, weights -- live in these three unfrozen
        # YAMLs. This hash makes a run auditable after the fact for exactly
        # which reward definitions produced it; editing a config changes it.
        "reward_config_sha256": {
            name: {"path": str(path), "sha256": reward_config_sha256[name]}
            for name, path in reward_config_paths_map.items()
        },
        # Adjudication (d): diagnostic only, never a success metric -- see
        # STRUCTURAL_METRICS / ARM_METRIC and the max_tanimoto_mean /
        # near_copy_fraction columns on every row.
        "drift_reference": {
            "path": str(drift_reference_path),
            "n_reference_fingerprints": similarity_monitor.n_reference,
            "has_similarity": similarity_monitor.has_similarity,
        },
        "rows": summary_rows,
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
                  f"margin={row.get('optimized_min_margin')} "
                  f"delta_ensemble={row.get('delta_ensemble')} "
                  f"ci=[{row.get('delta_ensemble_ci_low')}, "
                  f"{row.get('delta_ensemble_ci_high')}] "
                  f"sampling_matches_arm_b={row.get('sampling_matches_arm_b')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
