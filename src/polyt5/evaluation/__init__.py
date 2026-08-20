"""Evaluation layer for polyT5: the paper's filter cascade, metrics, reporting.

This package implements the *evaluation protocol* of Sahu et al. It sits on top
of :mod:`polyt5.chemistry`, which owns all SELFIES/RDKit handling, and adds the
parts that are specific to how the paper measures a model:

===================  ======================================================
Module               Responsibility
===================  ======================================================
``filters``          The nested SV -> TSD -> DD -> PV screen, including the
                     valency-1 terminus rule that PSMILES Validity requires.
``generation_metrics``  SR, SA statistics, novelty, diversity, and the TP
                     target-window metric, assembled into one report.
``regression_metrics``  MAE/RMSE/R^2/Pearson for the text-generating property
                     regressor, plus explicit accounting of non-numeric
                     outputs and averaging over the paper's five splits.
``similarity``       Loop-closed ECFP6 fingerprints and Tanimoto similarity.
``report``           Writing results into a run directory, and formatting a
                     console summary (which the *caller* prints).
===================  ======================================================

Three properties hold throughout, and the RL reward system depends on them:

* **No torch.** Nothing here imports torch, directly or transitively, so an
  evaluation or a reward computation runs with no GPU and no model in-process.
  A property predictor is *injected* as a callable rather than imported.
* **No printing.** Every entry point returns a structured dataclass or dict.
  Only :func:`format_console_summary` produces human-readable text, and it
  returns the string rather than writing it.
* **Total on adversarial input.** Garbage model output becomes a recorded
  failure, never an exception. A missing capability -- no SA scorer, no
  fingerprint support -- becomes ``None`` plus an explicit availability flag,
  never a substituted estimate.

Example:
    >>> from polyt5.evaluation import evaluate_generation, format_console_summary
    >>> report = evaluate_generation(["[At][C][C][At]", "garbage"])
    >>> report.counts.n_pv
    1
    >>> print(format_console_summary(report))  # doctest: +SKIP
"""

from __future__ import annotations

from .filters import (
    STAGE_DD,
    STAGE_PV,
    STAGE_SV,
    STAGE_TSD,
    STAGES,
    CandidateRecord,
    FilterCounts,
    apply_filter_cascade,
    has_valid_termini,
)
from .generation_metrics import (
    GenerationReport,
    diversity_metrics,
    evaluate_generation,
    target_property_rate,
)
from .regression_metrics import (
    METRIC_NAMES,
    RegressionReport,
    aggregate_over_splits,
    parse_numeric_predictions,
    regression_report,
)
from .report import format_console_summary, write_generation_report
from .similarity import (
    FINGERPRINTS_AVAILABLE,
    build_reference_fingerprints,
    ecfp6,
    loop_closed_mol,
    max_similarity_to_reference,
    mean_pairwise_tanimoto,
    tanimoto,
)
from .sweep import (
    DEFAULT_GRID,
    OBJECTIVES,
    PAPER_GRID,
    PAPER_SAMPLES_PER_CONFIG,
    PAPER_TEMPERATURES,
    PAPER_TOP_PS,
    SweepPoint,
    SweepResult,
    append_result_jsonl,
    read_results_jsonl,
    run_sweep_point,
    select_best,
    sweep_grid,
    sweep_to_dataframe,
    sweep_to_markdown,
)

__all__ = [
    "FINGERPRINTS_AVAILABLE",
    "METRIC_NAMES",
    "STAGES",
    "STAGE_DD",
    "STAGE_PV",
    "STAGE_SV",
    "STAGE_TSD",
    "CandidateRecord",
    "FilterCounts",
    "GenerationReport",
    "RegressionReport",
    "aggregate_over_splits",
    "apply_filter_cascade",
    "build_reference_fingerprints",
    "diversity_metrics",
    "ecfp6",
    "evaluate_generation",
    "format_console_summary",
    "has_valid_termini",
    "loop_closed_mol",
    "max_similarity_to_reference",
    "mean_pairwise_tanimoto",
    "parse_numeric_predictions",
    "regression_report",
    "tanimoto",
    "target_property_rate",
    "write_generation_report",
    "DEFAULT_GRID",
    "OBJECTIVES",
    "PAPER_GRID",
    "PAPER_SAMPLES_PER_CONFIG",
    "PAPER_TEMPERATURES",
    "PAPER_TOP_PS",
    "SweepPoint",
    "SweepResult",
    "append_result_jsonl",
    "read_results_jsonl",
    "run_sweep_point",
    "select_best",
    "sweep_grid",
    "sweep_to_dataframe",
    "sweep_to_markdown",
]
