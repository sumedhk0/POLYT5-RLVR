"""Persisting and formatting an evaluation run.

Two responsibilities, deliberately separated:

* :func:`write_generation_report` puts the machine-readable results into the
  run directory (``metrics.json`` and ``generations.jsonl``) through the
  :class:`polyt5.utils.RunDirectory` API, so every experiment lands in the
  layout described in ``docs/reproduction.md``.
* :func:`format_console_summary` *returns* a human-readable table. It does not
  print it. Library code in this repository never writes to stdout; the caller
  -- a script, a notebook, a training loop -- decides whether the string is
  printed, logged, or ignored.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .filters import CandidateRecord
from .generation_metrics import GenerationReport

if TYPE_CHECKING:  # import only for typing: keeps this module import-cheap
    from polyt5.utils.logging_utils import RunDirectory

__all__ = ["format_console_summary", "write_generation_report"]

METRICS_FILENAME = "metrics.json"
GENERATIONS_FILENAME = "generations.jsonl"


def write_generation_report(
    run_dir: RunDirectory,
    report: GenerationReport,
    *,
    generations: Sequence[CandidateRecord] | Sequence[dict[str, Any]] | None = None,
) -> None:
    """Write a generation evaluation into a run directory.

    Args:
        run_dir: The run directory to write into.
        report: The evaluation to persist. Its :meth:`GenerationReport.to_dict`
            summary goes to ``metrics.json``.
        generations: Per-candidate rows for ``generations.jsonl``. Accepts
            :class:`CandidateRecord` objects or plain dicts. When ``None`` or
            empty, no generations file is created -- an empty file would be
            indistinguishable from a run that produced nothing.

    Returns:
        ``None``. Paths are owned by ``run_dir``.
    """
    run_dir.write_json(METRICS_FILENAME, report.to_dict())

    if not generations:
        return
    rows = [
        row.to_dict() if isinstance(row, CandidateRecord) else dict(row)
        for row in generations
    ]
    run_dir.append_jsonl(GENERATIONS_FILENAME, rows)


def format_console_summary(report: GenerationReport) -> str:
    """Render an evaluation as a readable table.

    The four screening stages are shown in cascade order with both the absolute
    count and the fraction of the raw batch, because the filters are nested and
    a stage count alone is not interpretable without its denominator.

    Args:
        report: The evaluation to render.

    Returns:
        A multi-line string. The caller prints it; this function does not.
        Unavailable measurements are rendered as ``n/a`` and never as ``0``.
    """
    counts = report.counts
    lines: list[str] = [
        "Generation evaluation",
        "=" * 46,
        f"{'generated':<26}{counts.n_input:>10}",
        "",
        "Filter cascade (nested, SV > TSD > DD > PV)",
        "-" * 46,
        _stage_line("SV   valid structures", counts.n_sv, counts.sv_rate),
        _stage_line("TSD  not in training", counts.n_tsd, counts.tsd_rate),
        _stage_line("DD   unique in batch", counts.n_dd, counts.dd_rate),
        _stage_line("PV   two [At], valency 1", counts.n_pv, counts.pv_rate),
        "",
        "Quality",
        "-" * 46,
        _metric_line("SR   selfies round trip", report.sr_rate, percent=True),
        _metric_line("unique polymers", report.n_unique),
        _metric_line("novel polymers", report.n_novel),
        _metric_line("duplicate rate", report.duplicate_rate, percent=True),
    ]

    if report.sa_available:
        lines += [
            _metric_line("SA   mean", report.sa_mean),
            _metric_line("SA   median", report.sa_median),
            _metric_line("SA   fraction > 6", report.sa_fraction_above_6, percent=True),
        ]
    else:
        lines.append(f"{'SA   scorer':<26}{'unavailable':>10}")

    diversity = report.diversity or {}
    if report.fingerprints_available:
        lines.append(
            _metric_line("mean pairwise Tanimoto", diversity.get("mean_pairwise_tanimoto"))
        )
    else:
        lines.append(f"{'fingerprints':<26}{'unavailable':>10}")

    if report.n_property_values is not None:
        window = ""
        if report.property_target is not None and report.property_tolerance is not None:
            window = f" ({report.property_target:g} +/- {report.property_tolerance:g})"
        lines += [
            "",
            f"Predicted property{window}",
            "-" * 46,
            _metric_line("scored candidates", report.n_property_values),
            _metric_line("mean", report.property_mean),
            _metric_line("median", report.property_median),
            _metric_line("std", report.property_std),
            _metric_line("TP   in target window", report.target_property_rate, percent=True),
        ]

    return "\n".join(lines)


def _stage_line(label: str, count: int, rate: float) -> str:
    """Format one cascade stage as ``label  count  (pct of input)``."""
    return f"{label:<26}{count:>10}{rate * 100:>9.1f}%"


def _metric_line(label: str, value: Any, *, percent: bool = False) -> str:
    """Format one metric, rendering ``None`` as ``n/a``."""
    if value is None:
        return f"{label:<26}{'n/a':>10}"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{label:<26}{value:>10}"
    if percent:
        return f"{label:<26}{value * 100:>9.1f}%"
    return f"{label:<26}{value:>10.3f}"
