"""Arm-comparison matrix: Arm A/B (Phase 1/2, supervised) vs the Phase-3
GRPO/RLVR arms.

Phase 3. NOT part of the published polyT5 method - see ``docs/rlvr_plan.md``.

2026-08-23: Tg was dropped from every RLVR reward -- see
``polyt5.rewards.composite``'s module docstring. The current arm set is
``validity``/``novelty``/``synthesisability``/``composite``/``control``, all
fully verifiable; ``constraint`` is kept as a redefined multi-criterion
conjunction; ``accuracy`` (Task 6's original C1) is RETIRED but stays
reportable here because its round-1 run completed before the cut. See
:data:`RLVR_ARMS`.

For every arm this script SAMPLES FRESH candidates under the fixed evaluation
protocol recorded in ``artifacts/baseline/frozen_baseline.json``
(``evaluation_protocol``: targets 300/400/500 K, ``n_per_target`` each) and
scores them -- it never reuses the numbers already recorded there for Arm A/B.
That keeps every row of the matrix produced by the SAME measurement, which is
the only way the columns are actually comparable:

* **Arm A / Arm B** -- the frozen ``generation`` checkpoint (SHA-256 verified),
  resampled at the ``arm_a_default_sampling`` / ``arm_b_tuned_sampling``
  temperature and ``top_p`` recorded in the frozen baseline.
* **The RLVR arms** -- each arm's own GRPO-trained policy checkpoint
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

[DECISION] (Ruling D, constraint arm -- SUPERSEDED 2026-08-23) ``constraint``'s
optimized metric (``constraint_satisfaction_rate``) still gets both an
``_auditor`` and an ``_ensemble`` column, unlike ``pv_rate`` above, but the
MIXED-measurement reasoning this decision originally recorded no longer
applies: :class:`~polyt5.rewards.composite.ConstraintArm` was redefined the
same date to drop its Tg clause entirely (synthesisable AND novel only), so
it reads neither of ``predictions``'s Tg-carrying elements at all any more.
``constraint_satisfaction_rate_auditor`` and ``constraint_satisfaction_rate_
ensemble`` are therefore expected to be NUMERICALLY IDENTICAL for every row
-- both columns are kept, rather than collapsed into one structural column
like ``pv_rate``, purely so every arm's row has the same shape; a divergence
between them would itself be evidence the arm regained a predictor
dependency. ``novelty``, ``synthesisability`` and ``composite`` are the same
situation: none of the three reads a predictor triple either. See
:data:`SCORED_ARM_NAMES`, :class:`ArmScorers`, and each arm's own docstring
in :mod:`polyt5.rewards.composite`.

See ``ARM_METRIC`` for which column each arm is judged on and why. Every arm
in :data:`SCORED_ARM_NAMES` (``accuracy``, ``novelty``, ``synthesisability``,
``composite``, ``constraint``) needs PER-CANDIDATE scoring (not an aggregate
the paper's sweep machinery already returns), so this script reuses the
actual ``polyt5.rewards`` arm objects -- ``build_reward_arm(name, ...)`` for
each -- built ONCE from the CANONICAL ``configs/rl/*.yaml`` and applied
identically to every row, never from an RLVR arm's own (possibly
``--set``-overridden) training config. That is what "computed identically
for every row" requires: the formula must be the same across rows,
independent of how any one arm was actually trained.

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
metric DEFINITIONS -- ``sa_max``, ``novelty_index``, (composite) ``weights``,
plus the now-unused-but-still-present ``tolerance``/``sigma0``/
``sigma_unknown``/``min_coverage``/``window_tolerance`` -- live in
``configs/rl/<arm>.yaml`` for every arm in :data:`SCORED_ARM_NAMES`
(``accuracy``, ``novelty``, ``synthesisability``, ``composite``,
``constraint``), which are git-tracked but otherwise unfrozen. Editing one
silently changes what the pre-registered criterion MEANS, with no trace. This
script hashes exactly those five files (:func:`reward_config_paths`,
:func:`hash_reward_configs`) once, when it builds :class:`ArmScorers` from
them, and records the hash on every row (``*_reward_config_sha256``) and in
``summary.json`` (``reward_config_sha256``) -- so a run is auditable after the
fact for which reward definitions actually produced it. ``validity`` and
``control`` are not included: ``validity``'s optimized metric (``pv_rate``)
is structural RDKit chemistry with no ``reward:`` block in the loop at all,
and ``control`` optimizes nothing.

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
import logging
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: A plain ``logging.getLogger`` (propagate=True, no handlers of its own),
#: deliberately NOT ``polyt5.utils.get_logger`` -- that function is
#: idempotent per name and ``main()`` calls
#: ``get_logger("polyt5.compare_arms", log_file=...)`` to attach the run's
#: own file handler; a module-level call here under that SAME name would
#: run first (at import time), permanently pre-empt ``main()``'s handler
#: attachment (``get_logger`` returns the existing logger unchanged once it
#: already has handlers), and silently break ``compare_arms.log``. This
#: logger exists only for the rare, defensive warning inside
#: :func:`discover_seed_runs` (review finding 6).
_LOGGER = logging.getLogger(__name__)

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
    run_experiment_name,
    sha256_of_file,
    verify_artifact,
)

#: Every Phase-3 arm this script can score. 2026-08-23: Tg was dropped from
#: every RLVR reward (see ``polyt5.rewards.composite``'s module docstring).
#: ``accuracy`` is RETIRED -- its round-1 run completed BEFORE that cut and
#: is kept reportable for reproducibility, but is not part of the arm set a
#: new run should choose from. ``novelty`` and ``synthesisability`` are new;
#: ``composite`` and ``constraint`` keep their names but score REDEFINED,
#: Tg-free rewards -- neither has been trained under its new definition.
#: ``control`` (Addition 1 -- see ``polyt5.rewards.composite.ControlArm``)
#: has no entry in :data:`ARM_METRIC`: it optimizes nothing, so its matrix
#: row's ``success`` is always ``None``/N-A, exactly like arm_a/arm_b's.
RLVR_ARMS: tuple[str, ...] = (
    "accuracy", "validity", "novelty", "synthesisability", "composite", "constraint", "control",
)

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
#: ``polyt5.rewards.composite``. 2026-08-23: Tg dropped from every reward
#: below except the retired ``accuracy``; see that module's docstring.
#:   accuracy         - RETIRED. AccuracyArm's reward IS mean(closeness x
#:                      confidence) over the FULL batch, so it is scored
#:                      directly with the arm object -> accuracy_score
#:                      (higher). NOT property_mae; see the module docstring.
#:                      Kept only so the completed round-1 run stays
#:                      reportable.
#:   validity         - ValidityArm's reward IS "cleared the full
#:                      SV->TSD->DD->PV cascade" -> pv_rate (higher);
#:                      STRUCTURAL (see above).
#:   novelty          - NEW. NoveltyArm's reward is "cleared the gate AND
#:                      novel" -> novelty_rate (higher). No predictor
#:                      involved.
#:   synthesisability - NEW. SynthesisabilityArm's reward is "cleared the
#:                      gate AND SA <= sa_max" -> sa_pass_rate (higher). No
#:                      predictor involved.
#:   composite        - REDEFINED. CompositeArm's reward is now
#:                      w_pv*pv_pass + w_novelty*novel + w_sa*sa_pass +
#:                      w_diversity*first_occurrence -- no Tg term, no
#:                      predictor involved -- scored directly with the arm
#:                      object -> composite_score (higher). Never trained
#:                      under this definition.
#:   constraint       - REDEFINED. ConstraintArm's reward is now a
#:                      conjunction over (SA <= sa_max) AND novel -- the Tg
#:                      window and ensemble-backed clauses are gone, no
#:                      predictor involved -- scored directly ->
#:                      constraint_satisfaction_rate (higher). Never trained
#:                      under this definition.
#:
#: This mapping is ordinary mutable Python and therefore is NOT the
#: pre-registration. ``frozen_baseline.json``'s ``pre_registered_metrics`` is;
#: :func:`check_pre_registration` refuses to run if the two disagree.
ARM_METRIC: dict[str, tuple[str, str]] = {
    "accuracy": ("accuracy_score", "higher"),
    "validity": ("pv_rate", "higher"),
    "novelty": ("novelty_rate", "higher"),
    "synthesisability": ("sa_pass_rate", "higher"),
    "composite": ("composite_score", "higher"),
    "constraint": ("constraint_satisfaction_rate", "higher"),
}

#: Minimum improvement over arm_b, in each metric's own units, below which a
#: difference is not called a win no matter how tight the bootstrap interval
#: is. Pre-registered in ``frozen_baseline.json`` alongside the metric names.
ARM_MIN_MARGIN: dict[str, float] = {
    "accuracy": 0.01,
    "validity": 0.02,
    "novelty": 0.02,
    "synthesisability": 0.02,
    "composite": 0.02,
    "constraint": 0.02,
}

#: Bootstrap settings, pre-registered in ``frozen_baseline.json``'s
#: ``success_criterion_statistics``.
N_BOOTSTRAP = 2000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 0

#: Statistical properties of the ACROSS-SEED criterion (Addition 2),
#: pre-registered in ``frozen_baseline.json``'s ``success_criterion_
#: statistics`` alongside :data:`N_BOOTSTRAP` / :data:`BOOTSTRAP_CONFIDENCE`
#: / :data:`BOOTSTRAP_SEED` and checked the identical way by
#: :func:`check_pre_registration` -- see that function's docstring (review
#: finding 2: the across-seed rule used to be pre-registered PROSE with no
#: executable binding at all) and ``frozen_baseline.json``'s
#: ``success_criterion_across_seeds`` / ``success_criterion_statistics.
#: across_seed_rule`` for what these bind.
ACROSS_SEED_BOOTSTRAP = False
ACROSS_SEED_MIN_MARGIN_SOURCE = "same_as_arm_min_margin"
ACROSS_SEED_REQUIRES_UNANIMITY = True

#: ``max_tanimoto_mean`` / ``near_copy_fraction`` (adjudication (d)) and the
#: three ``*_reward_config_sha256`` columns (pre-registration integrity, see
#: the module docstring) are DIAGNOSTIC / provenance columns computed
#: identically for every row, arm_a's and arm_b's included -- they are never
#: read by :data:`ARM_METRIC` or :func:`apply_success_criterion`.
#:
#: ``seed`` (Addition 2: multi-seed support) is the training seed that
#: produced an RLVR row's checkpoint -- ``None`` for arm_a/arm_b, which are
#: not GRPO-trained runs at all. An arm can now have more than one row (one
#: per discovered seed under its run directory, see
#: :func:`discover_seed_runs`); :func:`compute_seed_aggregates` groups them
#: back together by ``arm`` and reports mean +/- sd across seeds, alongside
#: (not instead of) these existing per-run columns.
MATRIX_COLUMNS: tuple[str, ...] = (
    "arm", "kind", "checkpoint", "checkpoint_sha256", "seed", "temperature", "top_p",
    "sampling_matches_arm_b", "n_requested",
    "n_sv", "sv_rate", "n_tsd", "tsd_rate", "n_dd", "dd_rate", "n_pv", "pv_rate",
    "sr_rate", "duplicate_rate", "max_tanimoto_mean", "near_copy_fraction", "mean_length",
    "property_mean_auditor", "property_mae_auditor", "tp_rate_auditor",
    "property_mean_ensemble", "property_mae_ensemble", "tp_rate_ensemble",
    "accuracy_score_auditor", "accuracy_score_ensemble",
    "novelty_rate_auditor", "novelty_rate_ensemble",
    "sa_pass_rate_auditor", "sa_pass_rate_ensemble",
    "composite_score_auditor", "composite_score_ensemble",
    "constraint_satisfaction_rate_auditor", "constraint_satisfaction_rate_ensemble",
    "novelty_index_sha256",
    "accuracy_reward_config_sha256", "novelty_reward_config_sha256",
    "synthesisability_reward_config_sha256", "composite_reward_config_sha256",
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

# ---------------------------------------------------------------- multi-seed (Addition 2)
#
# Today each arm runs once and apply_success_criterion's bootstrap captures
# sampling noise WITHIN that one run's candidates -- it says nothing about
# RUN-TO-RUN variance from training the same arm at a different seed. The
# functions below add a second, independent axis of aggregation on top of
# (not instead of) the existing per-row bootstrap columns: group every row an
# arm has -- one per discovered training seed -- back together, and report
# each metric's mean +/- sample sd ACROSS those seeds, plus an across-seed
# success verdict that compares the arm's across-seed MEAN against arm_b by
# the same pre-registered min_margin. There is deliberately no bootstrap
# here: 1-3 independent training runs cannot support a second, seed-level
# bootstrap the way ~1500 candidates can support one over sampling noise.

#: Bookkeeping / identity / verdict columns excluded from cross-seed mean+-sd
#: aggregation (see :func:`compute_seed_aggregates`) -- everything else in
#: :data:`MATRIX_COLUMNS` is a per-row MEASURED quantity and gets aggregated.
#: Derived by exclusion, not a second hand-maintained list, for the same
#: reason :func:`skipped_arm_row` derives from :data:`MATRIX_COLUMNS`: a
#: hand-maintained list drifts the moment a column is added.
_SEED_AGGREGATE_EXCLUDE: frozenset[str] = frozenset({
    "arm", "kind", "checkpoint", "checkpoint_sha256", "seed", "temperature", "top_p",
    "sampling_matches_arm_b", "n_requested", "novelty_index_sha256",
    "accuracy_reward_config_sha256", "novelty_reward_config_sha256",
    "synthesisability_reward_config_sha256", "composite_reward_config_sha256",
    "constraint_reward_config_sha256", "optimized_metric", "optimized_min_margin",
    "optimized_value_auditor", "optimized_value_ensemble",
    "delta_ensemble", "delta_ensemble_ci_low", "delta_ensemble_ci_high",
    "delta_auditor", "delta_auditor_ci_low", "delta_auditor_ci_high",
    "beats_arm_b", "survives_audit", "success",
})

#: Every MATRIX_COLUMNS cell that is a measured quantity, in order --
#: :func:`compute_seed_aggregates` computes a cross-seed :func:`_seed_stat`
#: for each one. Deliberately not limited to an arm's own OPTIMIZED metric:
#: if the negative control (``control``, which optimizes nothing) shows
#: apparent improvement in ``pv_rate``, ``duplicate_rate``, or any other
#: column here, that is the diagnostic ControlArm's docstring describes.
SEED_AGGREGATE_METRICS: tuple[str, ...] = tuple(
    column for column in MATRIX_COLUMNS if column not in _SEED_AGGREGATE_EXCLUDE
)

#: Matches a run directory's ``_seed<N>`` suffix (see
#: ``train_grpo.run_experiment_name``).
#:
#: Review finding 6: deliberately rejects a leading zero (``0|[1-9]\d*``, not
#: bare ``\d+``) so ``_seed1`` and ``_seed01`` cannot both parse to the
#: integer key ``1`` and silently collide in :func:`discover_seed_runs`'s
#: ``found`` dict -- one of the two directories would disappear from the
#: aggregate with no trace. ``train_grpo.run_experiment_name`` builds this
#: suffix with an f-string int format (``f"_seed{seed}"``), which never
#: produces a leading zero, so a directory shaped like one can only be
#: hand-made; rejecting it here means it is simply not recognised as a seed
#: directory rather than risking that collision. The optional ``-`` keeps a
#: negative ``--set seed=-1`` run (reachable, though not via
#: ``run_round1.py``, which only ever passes the ints handed to ``--seeds``)
#: discoverable.
_SEED_SUFFIX_RE = re.compile(r"_seed(0|-?[1-9]\d*)$")


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

    2026-08-23: ``novelty`` and ``synthesisability`` joined ``accuracy``,
    ``composite`` and ``constraint`` here. Only ``accuracy``'s reward still
    reads a predictor triple at all -- the other four ignore ``predictions``
    entirely (see ``polyt5.rewards.composite``'s module docstring) -- so their
    auditor-scored and ensemble-scored columns are computed via two SEPARATE
    arm objects purely for uniformity with ``accuracy``'s row shape, and are
    expected to come out numerically IDENTICAL; a divergence between them
    would itself be evidence one of the four regained a predictor dependency.
    """

    accuracy: Any
    novelty: Any
    synthesisability: Any
    composite: Any
    constraint: Any


#: Every field name on :class:`ArmScorers`, in the order :func:`evaluate_arm`
#: scores them. A single source of truth for that iteration order, so the
#: dataclass and the loop over it cannot silently drift apart.
SCORED_ARM_NAMES: tuple[str, ...] = ("accuracy", "novelty", "synthesisability", "composite",
                                    "constraint")


def _resolve(path: str | Path) -> Path:
    """Resolve ``path`` relative to the repo root unless it is already absolute."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


#: The arms whose reward IS scored by an :class:`ArmScorers` object here --
#: exactly :data:`SCORED_ARM_NAMES`, and therefore also the arms whose
#: ``configs/rl/<arm>.yaml`` defines what its metric MEANS (``tolerance``,
#: ``sa_max``, ``novelty_index``, ``weights``, ...), hashed for provenance --
#: see :func:`reward_config_paths`. ``validity`` and ``control`` are
#: deliberately excluded: ``validity``'s optimized metric (``pv_rate``) is
#: structural RDKit chemistry computed directly from the screened batch, with
#: no ``reward:`` block or arm object involved at all -- see
#: :data:`STRUCTURAL_METRICS`; ``control`` optimizes nothing and has no entry
#: in :data:`ARM_METRIC`.
REWARD_CONFIG_ARMS: tuple[str, ...] = SCORED_ARM_NAMES


def reward_config_paths(repo_root: Path = REPO_ROOT) -> dict[str, Path]:
    """Canonical reward-definition YAMLs :func:`evaluate_arm`'s scorers are built from.

    [DECISION] These five files' ``sa_max``, ``novelty_index`` and (composite
    only) ``weights`` are exactly what ``accuracy_score`` / ``novelty_rate`` /
    ``sa_pass_rate`` / ``composite_score`` / ``constraint_satisfaction_rate``
    MEAN -- the pre-registered success metric for five of the six RLVR arms
    (``validity``'s ``pv_rate`` is structural; ``control`` optimizes nothing).
    Unlike ``frozen_baseline.json``, they are unhashed, unfrozen, git-tracked
    YAML: editing one silently changes what the pre-registered criterion
    measures, with no trace. Hashing them
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

    Review finding 2 (Addition 2): the across-seed criterion
    (:data:`ACROSS_SEED_BOOTSTRAP` / :data:`ACROSS_SEED_MIN_MARGIN_SOURCE` /
    :data:`ACROSS_SEED_REQUIRES_UNANIMITY`, applied by
    :func:`compute_seed_aggregates`) used to be pre-registered in
    ``frozen_baseline.json`` as PROSE -- ``success_criterion_across_seeds``
    and ``success_criterion_statistics.across_seed_rule`` -- with nothing in
    this function reading either key, which is the identical unbound-drift
    failure mode the rest of this docstring describes, just for a newer
    criterion. Both keys are now required to be present, and three
    structured statistics keys are checked against code the same way
    ``n_bootstrap``/``confidence``/``bootstrap_seed`` already are.

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
    # The three across-seed keys are REQUIRED, not merely compared-if-present.
    # A soft check lets a key be deleted to silently disable its own comparison,
    # which is precisely the drift this function exists to prevent -- and these
    # three define a criterion pre-registered before any second seed ran, so
    # there is no legacy record that could legitimately lack them. The first
    # three stay soft: records predating the bootstrap genuinely do not carry
    # them, and hard-requiring those would reject a valid older baseline.
    for key, coded_value, required in (
        ("n_bootstrap", N_BOOTSTRAP, False),
        ("confidence", BOOTSTRAP_CONFIDENCE, False),
        ("bootstrap_seed", BOOTSTRAP_SEED, False),
        ("across_seed_bootstrap", ACROSS_SEED_BOOTSTRAP, True),
        ("across_seed_min_margin_source", ACROSS_SEED_MIN_MARGIN_SOURCE, True),
        ("across_seed_requires_unanimity", ACROSS_SEED_REQUIRES_UNANIMITY, True),
    ):
        if key not in statistics:
            if required:
                problems.append(
                    f"{key} missing from frozen success_criterion_statistics -- the across-seed "
                    f"criterion is unregistered, and its absence must fail rather than skip"
                )
            continue
        if statistics[key] != coded_value:
            problems.append(f"{key}: code {coded_value!r} != frozen {statistics[key]!r}")

    # Review finding 2: presence, not just value-on-match -- a DELETED key
    # must fail too, not silently skip the comparison the way the six
    # optional value checks above do for a genuinely older record.
    if not frozen.get("success_criterion_across_seeds"):
        problems.append(
            "frozen_baseline.json has no success_criterion_across_seeds -- the across-seed "
            "criterion compute_seed_aggregates() applies is unregistered"
        )
    if not statistics.get("across_seed_rule"):
        problems.append(
            "frozen_baseline.json's success_criterion_statistics has no across_seed_rule -- "
            "the across-seed criterion compute_seed_aggregates() applies is unregistered"
        )
    return problems


def _latest_checkpoint(run_dir: Path) -> Path | None:
    """Return the highest-step checkpoint under a run directory, or ``None``."""
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return None
    candidates = sorted(checkpoints_dir.glob("step_*.pt"))
    return candidates[-1] if candidates else None


def resolve_arm_run_dir(
    arm_name: str, results_root: Path, *, seed: int = 0,
    repo_cfg: dict[str, Any] | None = None,
) -> Path:
    """The run directory ``train_grpo.py`` created for ``(arm_name, seed)``.

    Mirrors ``train_grpo.run_experiment_name`` exactly (seed 0 keeps the base
    name unsuffixed, matching every run directory that existed before
    multi-seed support did): the two must never drift apart, or this script
    would look in a directory ``train_grpo.py`` never wrote to.

    Args:
        arm_name: One of :data:`RLVR_ARMS`.
        results_root: Root results directory (``results/`` by default).
        seed: The training seed. ``0`` (the default) resolves to the exact
            same path this function always resolved to before multi-seed
            support existed.
        repo_cfg: The arm's own ``configs/rl/<arm>.yaml``, already loaded --
            passed by callers (:func:`discover_seed_runs`,
            :func:`load_rlvr_arm_model`) that already have it, so this
            function does not re-read the file on every seed. Loaded here
            when omitted.

    Returns:
        ``<results_root>/<base_name>`` (seed 0) or
        ``<results_root>/<base_name>_seed<N>`` (seed N), where ``base_name``
        is ``repo_cfg["experiment_name"]`` or ``f"grpo_{arm_name}"``.
    """
    if repo_cfg is None:
        repo_cfg = load_config(REPO_ROOT / "configs" / "rl" / f"{arm_name}.yaml")
    base_name = repo_cfg.get("experiment_name") or f"grpo_{arm_name}"
    return _resolve(results_root) / run_experiment_name(base_name, seed)


def discover_seed_runs(
    arm_name: str, results_root: Path, *, repo_cfg: dict[str, Any] | None = None,
) -> list[tuple[int, Path]]:
    """Every ``(seed, run_dir)`` pair this arm has a run directory for, sorted by seed.

    Backward compatible with a pre-multi-seed run: a directory with no
    ``_seedN`` suffix -- e.g. the in-flight ``results/grpo_accuracy/`` -- is
    discovered as seed 0, exactly where :func:`resolve_arm_run_dir` (seed 0)
    would look for it. An arm that was never trained at all returns ``[]``,
    not an error.

    Args:
        arm_name: One of :data:`RLVR_ARMS`.
        results_root: Root results directory (``results/`` by default).
        repo_cfg: The arm's own ``configs/rl/<arm>.yaml``, already loaded, or
            ``None`` to load it here.

    Returns:
        ``[(seed, run_dir), ...]`` sorted by seed ascending. A directory is
        included whether or not it actually holds a checkpoint yet --
        :func:`load_rlvr_arm_model` is what decides "not trained yet" and
        skips it; this function only reports what directories exist.
    """
    if repo_cfg is None:
        repo_cfg = load_config(REPO_ROOT / "configs" / "rl" / f"{arm_name}.yaml")
    base_name = repo_cfg.get("experiment_name") or f"grpo_{arm_name}"
    root = _resolve(results_root)

    found: dict[int, Path] = {}
    seed0_dir = root / base_name
    if seed0_dir.is_dir():
        found[0] = seed0_dir
    if root.is_dir():
        for candidate in root.glob(f"{base_name}_seed*"):
            if not candidate.is_dir():
                continue
            match = _SEED_SUFFIX_RE.search(candidate.name)
            if match is None:
                continue
            seed_value = int(match.group(1))
            if seed_value in found:
                # Should be unreachable for anything train_grpo.py itself
                # created (the regex above already forbids the one
                # collision -- a leading zero -- it could produce by
                # accident); still logged rather than silently dropped, so
                # a hand-made collision never loses a run with no trace.
                _LOGGER.warning(
                    "discover_seed_runs(%s): both %r and %r resolve to seed %d -- keeping %r, "
                    "dropping the other.", arm_name, found[seed_value].name, candidate.name,
                    seed_value, found[seed_value].name,
                )
                continue
            found[seed_value] = candidate
    return sorted(found.items())


def load_rlvr_arm_model(
    arm_name: str, results_root: Path, tokenizer, logger, *, seed: int = 0, device: Any = "cpu"
):
    """Load an RLVR arm's trained policy, plus the temperature/top_p it actually trained with.

    Args:
        arm_name: One of :data:`RLVR_ARMS`.
        results_root: Root results directory (``results/`` by default); the
            arm's run directory is looked up via :func:`resolve_arm_run_dir`.
        tokenizer: The tokenizer this comparison is scoring with -- the
            checkpoint's own recorded ``tokenizer_sha256`` is checked against
            it.
        logger: Where to report what was found or why an arm was skipped.
        seed: Which training seed's run directory to load (Addition 2).
            ``0`` (the default) resolves to exactly the same directory this
            function always loaded before multi-seed support existed.
        device: Torch device (or device string) the returned model is moved
            to before this function returns it -- the checkpoint itself is
            always read with ``map_location="cpu"``, so without this the
            model would silently stay on CPU regardless of what the rest of
            the run uses. Defaults to ``"cpu"`` (a no-op move) so existing
            callers that only care about the checkpoint contents, not device
            placement, are unaffected.

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
    run_dir = resolve_arm_run_dir(arm_name, results_root, seed=seed, repo_cfg=repo_cfg)

    checkpoint_path = _latest_checkpoint(run_dir)
    if checkpoint_path is None:
        logger.warning("arm %s seed %d: no checkpoint under %s -- skipping (not trained yet)",
                       arm_name, seed, run_dir / "checkpoints")
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
    model = model.to(device)

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
    run_seed: int | None = None,
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
        reward_config_sha256: ``{name: sha256, ...}`` for every arm in
            :data:`SCORED_ARM_NAMES` -- SHA-256 of the canonical
            ``configs/rl/*.yaml`` files ``auditor_scorers`` /
            ``ensemble_scorers`` were built from (:func:`hash_reward_configs`),
            recorded verbatim on every row for pre-registration provenance.
        run_seed: The training seed whose checkpoint ``model`` is (Addition
            2) -- recorded verbatim as the row's ``seed`` column, so
            :func:`compute_seed_aggregates` can group an arm's rows back
            together by seed. ``None`` (the default) for arm_a/arm_b, which
            are not GRPO-trained runs and therefore have no training seed.

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
    # exactly these. The auditor's are computed before the ensemble's for
    # every arm (see
    # test_evaluate_arm_auditor_predictions_are_wired_through_the_helper).
    #
    # 2026-08-23: only ``accuracy`` still reads a predictor triple at all --
    # ``novelty``/``synthesisability``/``composite``/``constraint`` ignore
    # ``predictions`` entirely (see ``polyt5.rewards.composite``'s module
    # docstring). They are still scored via BOTH scorer sets, for uniformity
    # with ``accuracy``'s row shape: the two columns are expected to come out
    # numerically identical, and a divergence would itself be evidence one of
    # the four regained a predictor dependency.
    values_auditor: dict[str, list[float]] = {}
    values_ensemble: dict[str, list[float]] = {}
    for name in SCORED_ARM_NAMES:
        values_auditor[name] = _arm_values(
            getattr(auditor_scorers, name), candidates, sample_targets, auditor_predictions)
        values_ensemble[name] = _arm_values(
            getattr(ensemble_scorers, name), candidates, sample_targets, ensemble_predictions)

    # pv_rate is structural: one 0/1 indicator per generated candidate, over
    # the full batch, whose mean IS batch.counts' pv_rate. Both predictor
    # slots share it, exactly as _metric_column already says.
    pv_indicators = [float(record.passed_pv) for record in batch.records]
    # property_mae's samples are per-SURVIVOR absolute errors -- the same
    # finite pairs property_columns averages -- which is precisely why it is
    # not any arm's success metric: its denominator moves with validity.
    mae_values_auditor = _abs_errors(auditor_screened, screened_targets)
    mae_values_ensemble = _abs_errors(ensemble_screened, screened_targets)

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
        "seed": run_seed,
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
        "accuracy_score_auditor": _mean(values_auditor["accuracy"]),
        "accuracy_score_ensemble": _mean(values_ensemble["accuracy"]),
        "novelty_rate_auditor": _mean(values_auditor["novelty"]),
        "novelty_rate_ensemble": _mean(values_ensemble["novelty"]),
        "sa_pass_rate_auditor": _mean(values_auditor["synthesisability"]),
        "sa_pass_rate_ensemble": _mean(values_ensemble["synthesisability"]),
        "composite_score_auditor": _mean(values_auditor["composite"]),
        "composite_score_ensemble": _mean(values_ensemble["composite"]),
        "constraint_satisfaction_rate_auditor": _mean(values_auditor["constraint"]),
        "constraint_satisfaction_rate_ensemble": _mean(values_ensemble["constraint"]),
        "novelty_index_sha256": novelty_index_sha256,
        "accuracy_reward_config_sha256": reward_config_sha256.get("accuracy"),
        "novelty_reward_config_sha256": reward_config_sha256.get("novelty"),
        "synthesisability_reward_config_sha256": reward_config_sha256.get("synthesisability"),
        "composite_reward_config_sha256": reward_config_sha256.get("composite"),
        "constraint_reward_config_sha256": reward_config_sha256.get("constraint"),
        SAMPLES_KEY: {
            "accuracy_score": {"auditor": values_auditor["accuracy"],
                               "ensemble": values_ensemble["accuracy"]},
            "novelty_rate": {"auditor": values_auditor["novelty"],
                             "ensemble": values_ensemble["novelty"]},
            "sa_pass_rate": {"auditor": values_auditor["synthesisability"],
                             "ensemble": values_ensemble["synthesisability"]},
            "composite_score": {"auditor": values_auditor["composite"],
                                "ensemble": values_ensemble["composite"]},
            "constraint_satisfaction_rate": {"auditor": values_auditor["constraint"],
                                             "ensemble": values_ensemble["constraint"]},
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


def _seed_stat(values: Sequence[Any]) -> dict[str, Any]:
    """Mean +/- sample sd of one metric's per-seed values (Addition 2).

    ``sd`` is ``None`` -- never a fabricated ``0.0`` -- whenever fewer than
    two finite values are available: a single training run has no
    run-to-run variance to report, and printing a spread of zero would claim
    a repeat-run agreement that was never measured. ``note`` states the seed
    count in words for exactly that case (``"1 seed"``/``"0 seeds"``), so a
    reader of the matrix cannot mistake "not measured" for "measured and
    found to be zero".

    Args:
        values: One value per seed, in whatever order the caller collected
            them (``None`` entries -- an arm's row missing this column --
            are dropped, not treated as ``0.0``).

    Returns:
        ``{"n_seeds", "mean", "sd", "note", "values"}``.
    """
    finite = [float(v) for v in values if v is not None]
    n = len(finite)
    if n == 0:
        return {"n_seeds": 0, "mean": None, "sd": None, "note": "0 seeds", "values": []}
    mean = _mean(finite)
    if n == 1:
        return {"n_seeds": 1, "mean": mean, "sd": None, "note": "1 seed", "values": finite}
    sd = float(np.std(np.asarray(finite, dtype=float), ddof=1))
    return {"n_seeds": n, "mean": mean, "sd": sd, "note": f"{n} seeds", "values": finite}


def _across_seed_clause(
    own_mean: float | None, base_value: Any, *, direction: str, min_margin: float,
) -> tuple[bool | None, float | None]:
    """One clause of the across-seed success criterion (Addition 2).

    Mirrors :func:`_clause`'s sign convention (positive delta always means
    "better than arm_b") without the bootstrap: an arm trained at 1-3 seeds
    has too few independent runs to support a second, seed-level bootstrap
    the way ~1500 candidates support one over sampling noise within a run
    (see ``frozen_baseline.json``'s ``success_criterion_across_seeds``).

    Args:
        own_mean: The arm's across-seed mean of one metric column (a
            :func:`_seed_stat` result's ``"mean"``), or ``None`` if unmeasured.
        base_value: arm_b's own (single-run) value of the SAME metric column.
        direction: ``"higher"`` or ``"lower"``.
        min_margin: The pre-registered minimum effect size.

    Returns:
        ``(verdict, delta)`` -- ``verdict`` is ``None``, not ``False``, when
        either side is unmeasured.
    """
    if own_mean is None or base_value is None:
        return None, None
    sign = 1.0 if direction == "higher" else -1.0
    delta = sign * (float(own_mean) - float(base_value))
    return bool(delta >= min_margin), delta


def _unanimous_clause(
    values: Sequence[float | None], base_value: Any, *, direction: str, min_margin: float,
) -> bool | None:
    """Whether EVERY individual seed value clears ``min_margin`` on its own.

    Review judgement on margin-only-on-the-mean: it can pass on seeds that
    disagree about the sign entirely -- e.g. per-seed deltas
    ``{+0.09, +0.01, -0.04}`` against ``min_margin=0.02`` give a mean of
    ``+0.02``, a pass, while one seed actually LOST to arm_b. Unanimity is
    distribution-free (no assumed sampling distribution, unlike a
    ``mean_delta - min_margin >= sd/sqrt(n)`` guard, which is not defensible
    at n=1-3), needs no minimum sample size, and is exactly what "not seed
    luck" means operationally. See ``frozen_baseline.json``'s
    ``success_criterion_across_seeds`` for the full reasoning and the two
    alternatives (a seed-level bootstrap, a sign test) considered and
    rejected.

    Args:
        values: One value per seed -- typically a :func:`_seed_stat`
            result's ``"values"`` (already ``None``-filtered).
        base_value: arm_b's own (single-run) value of the SAME metric column.
        direction: ``"higher"`` or ``"lower"``.
        min_margin: The pre-registered minimum effect size (same one
            :func:`_across_seed_clause` uses -- there is no separate
            unanimity margin).

    Returns:
        ``None`` -- not ``True`` -- when there are no values or arm_b's
        value is unmeasured: "not measured" and "measured and vacuously
        true over zero seeds" are different facts, and the latter would
        read as an unearned pass.
    """
    finite = [float(v) for v in values if v is not None]
    if not finite or base_value is None:
        return None
    sign = 1.0 if direction == "higher" else -1.0
    return all(sign * (value - float(base_value)) >= min_margin for value in finite)


def compute_seed_aggregates(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group RLVR rows by arm and report mean +/- sd across seeds, per metric.

    Complements, and does NOT replace, :func:`apply_success_criterion`'s
    per-row bootstrap-over-candidates clauses (still computed on every row,
    seed included -- see :data:`MATRIX_COLUMNS`' ``delta_*``/``beats_arm_b``/
    ``survives_audit``/``success`` columns): that bootstrap answers "is this
    ONE seed's candidate sample distinguishable from arm_b's", a question
    about SAMPLING noise within a single run. This function answers a
    different one -- "does the arm's own run-to-run variance across
    INDEPENDENT TRAINING SEEDS still show the improvement" -- which no
    amount of bootstrapping a single run's candidates can answer, since
    every candidate in that run shares the same trained checkpoint.

    Args:
        rows: Every row :func:`evaluate_arm`/:func:`skipped_arm_row`
            produced, arm_a's and arm_b's included. Only rows with
            ``kind == "rlvr"`` and an actual checkpoint (i.e. NOT a
            :func:`skipped_arm_row` placeholder, whose ``checkpoint`` is
            always ``None``) are grouped; an arm with zero trained seeds is
            simply absent from the result.

    Returns:
        Per arm: ``n_seeds``/``seeds`` (both derived from the SAME
        ``seed``-not-``None`` row list, so they cannot disagree -- review
        finding 7), ``n_seeds_matching_sampling``, ``sampling_mismatch_
        seeds`` / ``sampling_unknown_sampling_seeds`` (seeds excluded from
        the metrics below because ``sampling_matches_arm_b`` was ``False``
        or ``None`` respectively -- review finding 5 keeps those two facts
        distinct rather than collapsing both into one boolean),
        ``optimized_metric``, ``min_margin``, ``metrics`` (``{column:
        _seed_stat(...) for column in SEED_AGGREGATE_METRICS}``, computed
        ONLY over sampling-matching rows -- review finding 4: a seed
        excluded from the verdict must also be excluded from the published
        mean/sd, not silently averaged in), ``across_seed_beats_arm_b`` /
        ``across_seed_delta_ensemble`` / ``across_seed_survives_audit`` /
        ``across_seed_delta_auditor`` / ``across_seed_success`` (the
        mean-based verdict, unchanged in meaning), ``across_seed_unanimous_
        ensemble`` / ``across_seed_unanimous_auditor`` / ``across_seed_
        unanimous`` (whether EVERY individual matching seed clears
        ``min_margin`` on its own -- the review's statistical
        recommendation: mean-only can pass on seeds that disagree about the
        sign, see :func:`_unanimous_clause`), and ``combined_success`` (the
        actual conjunction of ``across_seed_success``, ``across_seed_
        unanimous`` and every contributing seed's own per-row bootstrap
        ``success`` -- review finding 8: ``success`` and ``across_seed_
        success`` used to be reported side by side with nothing joining
        them).

        Every verdict/unanimity/``combined_success`` key is ``None`` for an
        arm with no entry in :data:`ARM_METRIC` (``control`` -- see its
        docstring: optimizing nothing means nothing can beat or lose to
        arm_b) and ``None`` whenever any one of the arm's seeds was not
        confirmed sampled at arm_b's operating point (mirrors
        :func:`apply_success_criterion`'s own ``sampling_matches_arm_b``
        refusal, extended across every seed).
    """
    trained = [
        row for row in rows
        if row.get("kind") == "rlvr" and row.get("checkpoint") is not None
    ]
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in trained:
        by_arm.setdefault(row["arm"], []).append(row)

    arm_b = next((row for row in rows if row.get("arm") == "arm_b"), None)

    aggregates: dict[str, dict[str, Any]] = {}
    for arm_name, arm_rows in by_arm.items():
        # Finding 7: both derived from the SAME filtered list, so they
        # cannot disagree the way a bare len(arm_rows) vs a None-dropping
        # "seeds" list could.
        seeds = sorted(row["seed"] for row in arm_rows if row.get("seed") is not None)

        # Finding 5: apply_success_criterion already keeps "confirmed
        # mismatch" (False) and "unknown" (None) apart per row; this
        # preserves that distinction across the whole arm instead of
        # folding both into one collapsed boolean.
        mismatched_seeds = sorted(
            row["seed"] for row in arm_rows
            if row.get("sampling_matches_arm_b") is False and row.get("seed") is not None
        )
        unknown_sampling_seeds = sorted(
            row["seed"] for row in arm_rows
            if row.get("sampling_matches_arm_b") is None and row.get("seed") is not None
        )
        all_match = not mismatched_seeds and not unknown_sampling_seeds

        # Finding 4: a seed excluded from the verdict is excluded from the
        # published mean/sd too -- not averaged in and then separately
        # refused.
        matching_rows = [row for row in arm_rows if row.get("sampling_matches_arm_b") is True]
        metrics = {
            column: _seed_stat([row.get(column) for row in matching_rows])
            for column in SEED_AGGREGATE_METRICS
        }

        metric_name, direction = ARM_METRIC.get(arm_name, (None, None))
        entry: dict[str, Any] = {
            "n_seeds": len(seeds), "seeds": seeds,
            "n_seeds_matching_sampling": len(matching_rows),
            "sampling_mismatch_seeds": mismatched_seeds,
            "sampling_unknown_sampling_seeds": unknown_sampling_seeds,
            "metrics": metrics,
            "optimized_metric": metric_name, "min_margin": ARM_MIN_MARGIN.get(arm_name),
        }

        if metric_name is None or arm_b is None:
            entry["across_seed_beats_arm_b"] = None
            entry["across_seed_delta_ensemble"] = None
            entry["across_seed_survives_audit"] = None
            entry["across_seed_delta_auditor"] = None
            entry["across_seed_success"] = None
            entry["across_seed_unanimous_ensemble"] = None
            entry["across_seed_unanimous_auditor"] = None
            entry["across_seed_unanimous"] = None
            entry["combined_success"] = None
            aggregates[arm_name] = entry
            continue

        min_margin = float(ARM_MIN_MARGIN.get(arm_name, 0.0))

        ensemble_column = _metric_column(metric_name, "ensemble")
        ensemble_base = arm_b.get(ensemble_column)
        beats, delta_ensemble = _across_seed_clause(
            metrics[ensemble_column]["mean"], ensemble_base,
            direction=direction, min_margin=min_margin,
        )
        unanimous_ensemble = _unanimous_clause(
            metrics[ensemble_column]["values"], ensemble_base,
            direction=direction, min_margin=min_margin,
        )
        entry["across_seed_beats_arm_b"] = beats
        entry["across_seed_delta_ensemble"] = delta_ensemble
        entry["across_seed_unanimous_ensemble"] = unanimous_ensemble

        if metric_name in STRUCTURAL_METRICS:
            audit_clause: Any = True
            unanimous_auditor: Any = True
            entry["across_seed_survives_audit"] = STRUCTURAL_AUDIT_NOTE
            entry["across_seed_delta_auditor"] = None
            entry["across_seed_unanimous_auditor"] = STRUCTURAL_AUDIT_NOTE
        else:
            auditor_column = _metric_column(metric_name, "auditor")
            auditor_base = arm_b.get(auditor_column)
            audit_clause, delta_auditor = _across_seed_clause(
                metrics[auditor_column]["mean"], auditor_base,
                direction=direction, min_margin=min_margin,
            )
            unanimous_auditor = _unanimous_clause(
                metrics[auditor_column]["values"], auditor_base,
                direction=direction, min_margin=min_margin,
            )
            entry["across_seed_survives_audit"] = audit_clause
            entry["across_seed_delta_auditor"] = delta_auditor
            entry["across_seed_unanimous_auditor"] = unanimous_auditor

        if beats is None or audit_clause is None:
            entry["across_seed_success"] = None
        elif not all_match:
            # Refused, not failed -- mirrors apply_success_criterion's own
            # sampling_matches_arm_b handling: an incomparable operating
            # point says nothing about whether the arm worked.
            entry["across_seed_success"] = None
        else:
            entry["across_seed_success"] = bool(beats and audit_clause)

        if unanimous_ensemble is None or unanimous_auditor is None:
            entry["across_seed_unanimous"] = None
        elif not all_match:
            entry["across_seed_unanimous"] = None
        else:
            entry["across_seed_unanimous"] = bool(unanimous_ensemble and unanimous_auditor)

        # Finding 8: the actual conjunction -- every contributing seed's own
        # per-candidate bootstrap success, AND the across-seed mean, AND
        # across-seed unanimity. success and across_seed_success/across_
        # seed_unanimous remain separately reported (they answer different
        # questions), but a reader no longer has to perform this AND by hand.
        per_row_success = [row.get("success") for row in matching_rows]
        if (entry["across_seed_success"] is None or entry["across_seed_unanimous"] is None
                or not matching_rows or any(value is None for value in per_row_success)):
            entry["combined_success"] = None
        else:
            entry["combined_success"] = bool(
                entry["across_seed_success"] and entry["across_seed_unanimous"]
                and all(bool(value) for value in per_row_success)
            )

        aggregates[arm_name] = entry

    return aggregates


def format_seed_summary_markdown(aggregates: dict[str, dict[str, Any]]) -> str:
    """Render :func:`compute_seed_aggregates`' output as a markdown table.

    One row per RLVR arm with at least one trained seed, reporting its
    OPTIMIZED metric's across-seed mean and spread (ensemble and auditor,
    computed only over seeds sampled at arm_b's operating point -- review
    finding 4), the seed count, the mean-based verdict, whether EVERY
    individual seed unanimously clears the margin on its own (review's
    statistical recommendation -- see :func:`_unanimous_clause`), and the
    combined verdict (review finding 8). An arm with exactly one matching
    seed reports its spread as the literal ``"1 seed"``, never a fabricated
    ``0.0`` -- a single run has no run-to-run variance to report. ``control``
    (and any arm with no entry in :data:`ARM_METRIC`) reports ``"n/a"`` for
    the metric/verdict columns -- see :func:`compute_seed_aggregates`.

    Args:
        aggregates: :func:`compute_seed_aggregates`'s return value.

    Returns:
        Markdown text (a heading plus a table), or just the heading if
        ``aggregates`` is empty -- never raises on an empty input.
    """
    columns = (
        "arm", "optimized_metric", "n_seeds", "seeds", "mean_ensemble", "spread_ensemble",
        "mean_auditor", "spread_auditor", "across_seed_beats_arm_b", "across_seed_survives_audit",
        "across_seed_unanimous", "across_seed_success", "combined_success",
    )
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---:" for _ in columns) + "|"
    lines = ["", "### Cross-seed summary", "", header, sep]
    for arm_name, entry in aggregates.items():
        metric_name = entry.get("optimized_metric")
        ensemble_stat = (
            entry["metrics"].get(_metric_column(metric_name, "ensemble")) if metric_name else None
        )
        auditor_stat = (
            entry["metrics"].get(_metric_column(metric_name, "auditor")) if metric_name else None
        )
        seeds_cell = ",".join(str(s) for s in entry["seeds"]) if entry["seeds"] else "n/a"
        lines.append("| " + " | ".join([
            str(arm_name),
            metric_name or "n/a",
            str(entry["n_seeds"]),
            seeds_cell,
            _format_cell(ensemble_stat["mean"] if ensemble_stat else None),
            _format_seed_spread(ensemble_stat),
            _format_cell(auditor_stat["mean"] if auditor_stat else None),
            _format_seed_spread(auditor_stat),
            _format_cell(entry.get("across_seed_beats_arm_b")),
            _format_cell(entry.get("across_seed_survives_audit")),
            _format_cell(entry.get("across_seed_unanimous")),
            _format_cell(entry.get("across_seed_success")),
            _format_cell(entry.get("combined_success")),
        ]) + " |")
    return "\n".join(lines) + "\n"


def _format_seed_spread(stat: dict[str, Any] | None) -> str:
    """Render a :func:`_seed_stat`'s ``sd`` -- ``"1 seed"``, never a
    fabricated ``"0.0000"``, when fewer than two seeds were measured.
    """
    if stat is None:
        return "n/a"
    if stat["n_seeds"] <= 1:
        return stat["note"]
    return _format_cell(stat["sd"])


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
        # load_verified_model always loads to CPU (map_location="cpu") and
        # deliberately leaves the device decision to its caller -- train_grpo.py's
        # two callers rely on that and move policy/reference via GRPOTrainer /
        # ReferencePolicy. screen_candidates now also moves whatever model it is
        # handed onto its own device parameter (belt and suspenders), but this
        # loader is the one place in this script that owns "generation_model", so
        # it satisfies the contract itself rather than depending on that.
        generation_model = generation_model.to(device)
    except (FileNotFoundError, ValueError, KeyError) as error:
        logger.error("could not build the comparison: %s", error)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    logger.info("auditor=%s (held out of every reward path) ensemble=%s", auditor_key,
               frozen["reward_ensemble"])

    # accuracy_score / novelty_rate / sa_pass_rate / composite_score /
    # constraint_satisfaction_rate must be computed IDENTICALLY for every row
    # -- see the module docstring -- so the arm objects are built ONCE here,
    # from the canonical repo configs, and reused for every row including
    # arm_a and arm_b. Two sets, differing only in the ensemble_size that
    # describes the predictor whose triples they will score: the auditor is
    # one model (n=1 of 1 is full coverage), the reward ensemble is four
    # (n=1 of 4 means three members could not read the candidate, which is
    # the opposite situation -- see polyt5.rewards.tg). Only ``accuracy``'s
    # reward actually reads a triple; the other four ignore it entirely (see
    # SCORED_ARM_NAMES / ArmScorers above).
    ensemble_size = len(ensemble)
    # Pre-registration integrity: these five YAMLs are what accuracy_score /
    # novelty_rate / sa_pass_rate / composite_score /
    # constraint_satisfaction_rate MEAN, and they are unhashed, unfrozen,
    # git-tracked files -- see reward_config_paths's docstring. Hashed here,
    # ONCE, alongside loading them, so the hash on every row is provably the
    # hash of the file the scorers below were actually built from.
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
            for name in SCORED_ARM_NAMES
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
        # Addition 2: an arm can now have more than one trained seed under
        # its base run directory -- discover every one of them (seed 0
        # included, suffix-less, for backward compatibility with runs
        # started before multi-seed support existed) and evaluate each as
        # its own row, rather than only ever looking at a single directory.
        arm_repo_cfg = load_config(REPO_ROOT / "configs" / "rl" / f"{arm_name}.yaml")
        seed_runs = discover_seed_runs(arm_name, args.results_root, repo_cfg=arm_repo_cfg)
        if not seed_runs:
            logger.warning("arm %s: no run directory found under %s (any seed) -- skipping",
                           arm_name, args.results_root)
            rows.append(skipped_arm_row(arm_name))
            continue

        arm_rows: list[dict[str, Any]] = []
        for run_seed, _run_dir in seed_runs:
            loaded = load_rlvr_arm_model(
                arm_name, args.results_root, tokenizer, logger, seed=run_seed, device=device)
            if loaded is None:
                continue
            model, checkpoint_path, checkpoint_sha256, point = loaded
            label = f"Arm {arm_name}" if run_seed == 0 else f"Arm {arm_name} (seed {run_seed})"
            row = evaluate_arm(
                model, tokenizer, arm_key=arm_name, label=label, kind="rlvr",
                point=point, targets=targets, n_samples=n_samples,
                training_index=training_index,
                auditor=auditor, ensemble=ensemble, auditor_scorers=auditor_scorers,
                ensemble_scorers=ensemble_scorers, similarity_monitor=similarity_monitor,
                device=device, batch_size=args.batch_size,
                max_length=args.max_length, seed=args.seed, tolerance=tolerance,
                checkpoint_label=str(checkpoint_path), checkpoint_sha256=checkpoint_sha256,
                novelty_index_sha256=novelty_index_sha256,
                reward_config_sha256=reward_config_sha256,
                run_seed=run_seed,
            )
            arm_rows.append(row)

        if not arm_rows:
            rows.append(skipped_arm_row(arm_name))
            continue
        rows.extend(arm_rows)

    apply_success_criterion(rows)
    # Addition 2: a SECOND, independent axis of aggregation on top of (not
    # instead of) the per-row bootstrap apply_success_criterion just filled
    # in -- see compute_seed_aggregates' own docstring for why the two ask
    # different questions.
    seed_aggregates = compute_seed_aggregates(rows)

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
    if seed_aggregates:
        with md_path.open("a", encoding="utf-8") as handle:
            handle.write(format_seed_summary_markdown(seed_aggregates))
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
        # Addition 2: a second, independent axis of aggregation across
        # INDEPENDENT TRAINING SEEDS, alongside (not instead of) the
        # per-candidate bootstrap above -- see compute_seed_aggregates' own
        # docstring, and frozen_baseline.json's success_criterion_across_seeds
        # for the pre-registered rule. Each arm entry's "n_seeds" is the seed
        # count this run was requested to record.
        "success_criterion_across_seeds": frozen.get("success_criterion_across_seeds"),
        "seed_aggregates": seed_aggregates,
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
        # accuracy_score / novelty_rate / sa_pass_rate / composite_score /
        # constraint_satisfaction_rate's DEFINITIONS -- sa_max,
        # novelty_index, weights, plus the now-unused tolerance/sigma0/
        # sigma_unknown/min_coverage/window_tolerance -- live in these five
        # unfrozen YAMLs. This hash makes a run auditable after the fact for
        # exactly which reward definitions produced it; editing a config
        # changes it.
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
    for arm_name, entry in seed_aggregates.items():
        metric_name = entry.get("optimized_metric")
        ensemble_col = _metric_column(metric_name, "ensemble") if metric_name else None
        stat = entry["metrics"].get(ensemble_col) if ensemble_col else None
        spread_note = stat["note"] if stat else "n/a"
        print(f"  {arm_name:<12} across_seed_success={entry.get('across_seed_success')} "
              f"n_seeds={entry['n_seeds']} seeds={entry['seeds']} "
              f"spread=[{spread_note}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
