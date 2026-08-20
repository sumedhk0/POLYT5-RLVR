"""Generation-quality metrics: the screening yield, SR, SA, novelty, and TP.

This module assembles one :class:`GenerationReport` from a batch of generated
PSELFIES strings. It sits above :mod:`polyt5.evaluation.filters`, which does the
paper's SV/TSD/DD/PV screen, and above :mod:`polyt5.chemistry`, which owns all
SELFIES/RDKit handling.

Property statistics are produced only when the caller *injects* a predictor::

    report = evaluate_generation(candidates, target_property=500.0,
                                 property_predictor=my_model.predict)

The predictor is a plain ``Callable[[Sequence[str]], Sequence[float]]``, so this
module never imports the model -- and, transitively, never imports torch. That
keeps the evaluation layer importable inside an RL reward worker with no GPU and
no model in the process. When no predictor is supplied, every property statistic
is ``None``. None of them is ever estimated.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from polyt5.chemistry import SA_AVAILABLE, NoveltyIndex

from .filters import (
    SA_HARD_THRESHOLD,
    CandidateRecord,
    FilterCounts,
    apply_filter_cascade,
)
from .similarity import FINGERPRINTS_AVAILABLE, mean_pairwise_tanimoto

__all__ = [
    "GenerationReport",
    "diversity_metrics",
    "evaluate_generation",
    "target_property_rate",
]

#: Default cap on how many polymers enter the O(n^2) pairwise-similarity step.
DEFAULT_MAX_PAIRWISE = 200


@dataclass(frozen=True)
class GenerationReport:
    """Everything measurable about one batch of generated polymers.

    Attributes:
        counts: The nested SV/TSD/DD/PV screening counts.
        sr_rate: SELFIES Reproducibility -- the fraction of *all* generated
            strings that survive ``encoder(decoder(s)) == s``. The denominator
            is the raw batch, not the screened subset, matching the paper's
            "fraction of generated SELFIES strings".
        n_unique: Distinct polymers (by canonical PSMILES) among the SV
            survivors.
        n_novel: Of those distinct polymers, how many are absent from the
            training set. With no training index supplied the reference set is
            empty, so this equals ``n_unique``.
        duplicate_rate: Fraction of SV survivors that repeat a polymer already
            seen earlier in the batch, i.e. ``1 - n_unique / n_sv``.
        sa_available: Whether an SA scorer could be loaded at all.
        n_sa_scored: How many candidates actually received an SA score.
        sa_mean: Mean SA score over the screened candidates, or ``None``.
        sa_median: Median SA score, or ``None``.
        sa_fraction_above_6: Fraction scoring above the paper's threshold of 6,
            above which a candidate is structurally awkward. ``None`` when no
            candidate was scored -- never ``0.0``, which would claim that none
            were awkward.
        property_target: The TP target, or ``None`` when no predictor ran.
        property_tolerance: The TP tolerance, or ``None``.
        n_property_values: How many finite predictions came back, or ``None``.
        property_mean: Mean predicted property, or ``None``.
        property_median: Median predicted property, or ``None``.
        property_std: Standard deviation of the predictions, or ``None``.
        target_property_rate: The paper's TP metric -- the proportion of
            candidates whose predicted property lies within
            ``target +- tolerance`` -- or ``None``.
        fingerprints_available: Whether ECFP fingerprints work in this process.
        diversity: Output of :func:`diversity_metrics` over the screened set.
        records: One :class:`CandidateRecord` per input, in input order.
        screened_psmiles: Canonical PSMILES of the candidates that survived the
            whole cascade, in input order.
    """

    counts: FilterCounts
    sr_rate: float
    n_unique: int
    n_novel: int
    duplicate_rate: float

    sa_available: bool
    n_sa_scored: int
    sa_mean: float | None
    sa_median: float | None
    sa_fraction_above_6: float | None

    property_target: float | None
    property_tolerance: float | None
    n_property_values: int | None
    property_mean: float | None
    property_median: float | None
    property_std: float | None
    target_property_rate: float | None

    fingerprints_available: bool
    diversity: dict[str, Any]
    records: list[CandidateRecord] = field(default_factory=list, repr=False)
    screened_psmiles: list[str] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary.

        The per-candidate ``records`` are excluded: they belong in
        ``generations.jsonl``, not in a metrics summary.
        """
        return {
            "counts": self.counts.to_dict(),
            "sr_rate": self.sr_rate,
            "n_unique": self.n_unique,
            "n_novel": self.n_novel,
            "duplicate_rate": self.duplicate_rate,
            "sa_available": self.sa_available,
            "n_sa_scored": self.n_sa_scored,
            "sa_mean": self.sa_mean,
            "sa_median": self.sa_median,
            "sa_fraction_above_6": self.sa_fraction_above_6,
            "property_target": self.property_target,
            "property_tolerance": self.property_tolerance,
            "n_property_values": self.n_property_values,
            "property_mean": self.property_mean,
            "property_median": self.property_median,
            "property_std": self.property_std,
            "target_property_rate": self.target_property_rate,
            "fingerprints_available": self.fingerprints_available,
            "diversity": self.diversity,
        }


def target_property_rate(
    values: Sequence[Any], target: float, tolerance: float
) -> float:
    """The paper's TP metric: proportion of candidates inside a property window.

    Figure S10 reports the "proportion of candidates with predicted Tg within
    500 +- 50 K"; this generalises that to any ``(target, tolerance)``.

    Args:
        values: Predicted property values.
        target: Centre of the acceptance window.
        tolerance: Half-width of the window. Boundaries are inclusive.

    Returns:
        The fraction of values satisfying ``|v - target| <= tolerance``, or
        ``0.0`` when there is nothing to measure.

    Note:
        Non-finite entries (``None``, ``NaN``, ``inf``) are excluded from both
        numerator and denominator rather than counted as misses: they mean the
        predictor produced no answer, which is not the same as producing an
        out-of-range answer.
        # [AMBIGUITY] the paper assumes every candidate gets a prediction and so
        # does not distinguish these cases.
    """
    finite: list[float] = []
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(as_float):
            finite.append(as_float)

    if not finite:
        return 0.0
    hits = sum(1 for value in finite if abs(value - target) <= tolerance)
    return hits / len(finite)


def diversity_metrics(
    canonical_psmiles: Sequence[str],
    *,
    max_pairwise: int = DEFAULT_MAX_PAIRWISE,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarise how varied a set of generated polymers is.

    Args:
        canonical_psmiles: Canonical PSMILES of the polymers to summarise.
        max_pairwise: Cap on how many *distinct* polymers enter the pairwise
            Tanimoto step, which is O(n^2). Above the cap a reproducible random
            subsample is used, and ``n_pairwise_sampled`` records the size that
            was actually compared.
        seed: Seed for that subsample, so the number is reproducible.

    Returns:
        A JSON-serialisable dict with ``n_total``, ``n_unique``,
        ``unique_fraction``, ``most_common_count``, ``most_common_fraction``,
        ``mean_pairwise_tanimoto`` (``None`` when unavailable),
        ``n_pairwise_sampled`` and ``pairwise_sample_cap``.
    """
    entries = [p for p in canonical_psmiles if isinstance(p, str) and p]
    n_total = len(entries)

    tally: dict[str, int] = {}
    for entry in entries:
        tally[entry] = tally.get(entry, 0) + 1
    n_unique = len(tally)
    most_common_count = max(tally.values()) if tally else 0

    mean_similarity: float | None = None
    n_sampled = 0
    if FINGERPRINTS_AVAILABLE and n_unique >= 2:
        mean_similarity, n_sampled = mean_pairwise_tanimoto(
            list(tally), max_pairwise=max_pairwise, seed=seed
        )

    return {
        "n_total": n_total,
        "n_unique": n_unique,
        "unique_fraction": (n_unique / n_total) if n_total else 0.0,
        "most_common_count": most_common_count,
        "most_common_fraction": (most_common_count / n_total) if n_total else 0.0,
        "mean_pairwise_tanimoto": mean_similarity,
        "n_pairwise_sampled": n_sampled,
        "pairwise_sample_cap": max_pairwise,
        "fingerprints_available": FINGERPRINTS_AVAILABLE,
    }


def evaluate_generation(
    generated_pselfies: Sequence[str],
    *,
    training_index: NoveltyIndex | None = None,
    target_property: float | None = None,
    tolerance: float = 50.0,
    property_predictor: Callable[[Sequence[str]], Sequence[float]] | None = None,
    expected_termini: int = 2,
    compute_sa: bool = True,
    max_pairwise: int = DEFAULT_MAX_PAIRWISE,
    seed: int = 0,
) -> GenerationReport:
    """Run the full generation evaluation over one batch of model output.

    Args:
        generated_pselfies: Raw model outputs as PSELFIES strings.
        training_index: Known training polymers, for the TSD stage and for
            novelty. ``None`` means no reference set is available.
        target_property: Centre of the TP acceptance window, e.g. ``500.0`` K
            for the paper's Tg screen. Only used when a predictor is supplied.
        tolerance: Half-width of that window; the paper uses ``50.0``.
        property_predictor: Injected callable mapping screened PSMILES to
            predicted property values. Keeping it an argument is what lets this
            module stay free of any model import. When ``None``, every property
            statistic in the report is ``None``.
        expected_termini: Chain ends a well-formed repeat unit must have.
        compute_sa: Compute synthetic accessibility for screened candidates.
        max_pairwise: Cap for the O(n^2) pairwise diversity comparison.
        seed: Seed for the diversity subsample.

    Returns:
        A :class:`GenerationReport`. This function never raises on model output,
        and a predictor that raises is reported as "no property statistics"
        rather than being allowed to abort the evaluation.
    """
    records, counts = apply_filter_cascade(
        generated_pselfies,
        training_index=training_index,
        expected_termini=expected_termini,
        compute_sa=compute_sa,
    )

    n_reproducible = sum(1 for record in records if record.reproducible)
    sr_rate = (n_reproducible / counts.n_input) if counts.n_input else 0.0

    # Uniqueness and novelty are judged over the RDKit-valid candidates: an
    # unparseable string has no canonical form to compare, and the paper's
    # novelty claims are about structures, not strings.
    valid_canonical = [
        record.canonical_psmiles
        for record in records
        if record.passed_sv and record.canonical_psmiles is not None
    ]
    distinct = list(dict.fromkeys(valid_canonical))
    n_unique = len(distinct)
    duplicate_rate = (1.0 - n_unique / counts.n_sv) if counts.n_sv else 0.0
    if training_index is None:
        n_novel = n_unique
    else:
        n_novel = sum(1 for psmiles in distinct if psmiles not in training_index)

    screened = [
        record.canonical_psmiles
        for record in records
        if record.passed_pv and record.canonical_psmiles is not None
    ]

    sa_scores = [
        record.sa_score for record in records if record.sa_score is not None
    ]
    sa_mean = float(statistics.fmean(sa_scores)) if sa_scores else None
    sa_median = float(statistics.median(sa_scores)) if sa_scores else None
    sa_fraction_above_6 = (
        sum(1 for score in sa_scores if score > SA_HARD_THRESHOLD) / len(sa_scores)
        if sa_scores
        else None
    )

    (
        n_property_values,
        property_mean,
        property_median,
        property_std,
        tp_rate,
        used_target,
        used_tolerance,
    ) = _property_statistics(screened, property_predictor, target_property, tolerance)

    return GenerationReport(
        counts=counts,
        sr_rate=sr_rate,
        n_unique=n_unique,
        n_novel=n_novel,
        duplicate_rate=duplicate_rate,
        sa_available=SA_AVAILABLE,
        n_sa_scored=len(sa_scores),
        sa_mean=sa_mean,
        sa_median=sa_median,
        sa_fraction_above_6=sa_fraction_above_6,
        property_target=used_target,
        property_tolerance=used_tolerance,
        n_property_values=n_property_values,
        property_mean=property_mean,
        property_median=property_median,
        property_std=property_std,
        target_property_rate=tp_rate,
        fingerprints_available=FINGERPRINTS_AVAILABLE,
        diversity=diversity_metrics(screened, max_pairwise=max_pairwise, seed=seed),
        records=records,
        screened_psmiles=screened,
    )


def _property_statistics(
    screened: Sequence[str],
    predictor: Callable[[Sequence[str]], Sequence[float]] | None,
    target: float | None,
    tolerance: float,
) -> tuple[
    int | None, float | None, float | None, float | None, float | None, float | None, float | None
]:
    """Run the injected predictor and summarise its output.

    Args:
        screened: Canonical PSMILES that survived the screen.
        predictor: The injected callable, or ``None``.
        target: TP window centre, or ``None``.
        tolerance: TP window half-width.

    Returns:
        ``(n_values, mean, median, std, tp_rate, target, tolerance)``, all
        ``None`` when no predictor was supplied or the predictor failed. The
        TP rate is additionally ``None`` when no target was given, since a
        window with no centre is meaningless.
    """
    nothing = (None,) * 7
    if predictor is None:
        return nothing  # type: ignore[return-value]

    try:
        raw_values = list(predictor(screened))
    except Exception:
        # An injected predictor is caller code; a failure there must not turn a
        # generation evaluation into a crash, and must not be reported as zero.
        return nothing  # type: ignore[return-value]

    finite: list[float] = []
    for value in raw_values:
        if value is None or isinstance(value, bool):
            continue
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(as_float):
            finite.append(as_float)

    if not finite:
        return (0, None, None, None, None, target, tolerance if target is not None else None)

    mean = float(statistics.fmean(finite))
    median = float(statistics.median(finite))
    std = float(statistics.pstdev(finite)) if len(finite) > 1 else 0.0
    tp_rate = target_property_rate(finite, target, tolerance) if target is not None else None
    return (
        len(finite),
        mean,
        median,
        std,
        tp_rate,
        target,
        tolerance if target is not None else None,
    )
