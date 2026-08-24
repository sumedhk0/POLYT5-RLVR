"""The sampling-hyperparameter sweep: Arm B of the eventual RLVR comparison.

Why this module exists
======================
The research question is whether GRPO/RLVR beats supervised polyT5 at
property-conditioned generation. That comparison is only honest against a
*properly tuned* supervised baseline, because much of what RL appears to buy
can often be had by turning the sampling temperature down::

    Arm A   supervised polyT5 + default sampling
    Arm B   supervised polyT5 + sampling tuned on validation   <- this module
    Arm C   supervised polyT5 + GRPO/RLVR

It is also a direct reproduction of the paper's own sweep (Sahu et al., npj
Artificial Intelligence 2026): ``top_p`` in {0.75, 0.95}, temperature
0.1 -> 2.0 in increments of 0.1, fine-tuning epochs 1 -> 15, 10,000 polymers
per configuration -- 3 model sizes x 15 epochs x 20 temperatures x 2 top_p =
1,800 configurations, over which 6,171,066 candidates passed the PV filter.
The paper reports optima at ``(epochs, temperature, top_p)`` of (14, 0.9, 0.75)
for small, (6, 1.1, 0.75) for medium and (8, 1.1, 0.95) for large.

The paper publishes its per-configuration counts ONLY as unlabeled heatmaps.
There is no numeric table to compare against, so :func:`sweep_to_markdown`
produces the numbers the paper does not.

The trade-off being measured
============================
The paper states it verbatim: a higher ``top_p`` gives "greater chemical
diversity but also a higher fraction of invalid structures", a lower ``top_p``
gives "reduced invalid outputs but increasing duplication"; and more
fine-tuning epochs first reduces invalid candidates and then increases
duplicates. That is why :func:`select_best` takes an explicit ``objective``:
optimising the PV yield alone drives the sweep towards cold, repetitive
sampling, which is precisely the failure mode Arm B exists to expose.

Design constraints inherited from :mod:`polyt5.evaluation`
==========================================================
* **No printing.** Every entry point returns a structured value; the script in
  ``scripts/sampling_sweep.py`` does the printing.
* **Torch stays optional at import time.** ``torch`` is imported inside the two
  functions that actually decode, so importing this module -- and therefore the
  whole ``polyt5.evaluation`` package -- still costs no torch. Grid building,
  objective selection and tabulation all work in a GPU-free reward worker.
* **The property predictor is injected**, never imported, exactly as in
  :func:`polyt5.evaluation.evaluate_generation`. That keeps the sweep usable
  with no property model at all and preserves the seam the RL phase needs.
* **Total on adversarial model output.** Garbage decodes become recorded
  failures in the cascade, never exceptions.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from polyt5.chemistry import NoveltyIndex
from polyt5.data.prepare import format_property_value

from .filters import CandidateRecord, FilterCounts, apply_filter_cascade
from .generation_metrics import target_property_rate

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "COUNT_KEYS",
    "DEFAULT_GRID",
    "DEFAULT_TEMPERATURES",
    "DEFAULT_TOP_PS",
    "OBJECTIVES",
    "PAPER_CONFIGURATIONS",
    "PAPER_EPOCHS",
    "PAPER_GRID",
    "PAPER_SAMPLES_PER_CONFIG",
    "PAPER_TEMPERATURES",
    "PAPER_TOP_PS",
    "SWEEP_COLUMNS",
    "ScreenedBatch",
    "SweepPoint",
    "SweepResult",
    "append_result_jsonl",
    "assign_targets",
    "generate_candidates",
    "property_columns",
    "read_results_jsonl",
    "run_sweep_point",
    "screen_candidates",
    "select_best",
    "sweep_grid",
    "sweep_to_dataframe",
    "sweep_to_markdown",
]

# One SELFIES token is one bracket group, e.g. "[C]", "[=Branch1]", "[At]".
_SELFIES_TOKEN_RE = re.compile(r"\[[^\]]+\]")


class _Tokenizer(Protocol):
    """Duck type of :class:`polyt5.tokenization.PolyT5Tokenizer`.

    Only the four members the sweep actually uses are required, so a test can
    substitute a stub and this module never has to import the tokenizer.
    """

    pad_id: int
    eos_id: int
    decoder_start_token_id: int

    def batch_encode(self, texts: Iterable[str], **kwargs: Any) -> dict[str, list[list[int]]]:
        """Encode a batch of strings to padded id rows."""
        ...  # pragma: no cover - protocol

    def batch_decode(self, id_lists: Sequence[Sequence[int]], **kwargs: Any) -> list[str]:
        """Decode id rows back to strings."""
        ...  # pragma: no cover - protocol


# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepPoint:
    """One configuration of the sweep.

    Attributes:
        temperature: Sampling temperature. The paper sweeps 0.1 -> 2.0.
        top_p: Nucleus mass. The paper uses {0.75, 0.95}.
        epoch: Fine-tuning epoch this point is evaluated at, or ``None`` when
            the sweep runs against a single fixed checkpoint. The paper sweeps
            1 -> 15 and treats the epoch as a genuine sampling-side axis,
            because the amount of fine-tuning trades invalid candidates against
            duplicates in the same way temperature does.
        checkpoint: Path of the checkpoint to load for this point, or ``None``
            to reuse whatever model the caller passes in.
    """

    temperature: float
    top_p: float
    epoch: int | None = None
    checkpoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "epoch": self.epoch,
            "checkpoint": self.checkpoint,
        }


def _validate_axes(temperatures: Sequence[float], top_ps: Sequence[float]) -> None:
    """Reject malformed sweep axes.

    Args:
        temperatures: Candidate temperature axis.
        top_ps: Candidate nucleus-mass axis.

    Raises:
        ValueError: If an axis is empty, contains duplicates, or contains a
            value the decoder would reject.
    """
    if not temperatures:
        raise ValueError("temperatures must not be empty.")
    if not top_ps:
        raise ValueError("top_ps must not be empty.")
    if len(set(temperatures)) != len(temperatures):
        raise ValueError(f"duplicate temperatures in axis: {list(temperatures)}")
    if len(set(top_ps)) != len(top_ps):
        raise ValueError(f"duplicate top_p values in axis: {list(top_ps)}")
    for temperature in temperatures:
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {temperature}.")
    for top_p in top_ps:
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}.")


def sweep_grid(
    *,
    temperatures: Sequence[float],
    top_ps: Sequence[float],
    epochs: Sequence[int] | None = None,
    checkpoints: Mapping[int, str] | None = None,
) -> list[SweepPoint]:
    """Build the full cartesian grid of sweep points.

    The iteration order is epoch-outermost, then ``top_p``, then temperature,
    so a run that walks the returned list in order changes checkpoints as
    rarely as possible -- reloading a checkpoint is the only expensive axis.

    Args:
        temperatures: Temperature axis; must be non-empty and duplicate-free.
        top_ps: Nucleus-mass axis; must be non-empty, duplicate-free, and in
            ``(0, 1]``.
        epochs: Fine-tuning-epoch axis. ``None`` means "one fixed checkpoint",
            and every returned point has ``epoch=None``.
        checkpoints: Optional ``epoch -> checkpoint path`` mapping. Epochs with
            no entry get ``checkpoint=None``.

    Returns:
        ``len(temperatures) * len(top_ps) * max(len(epochs), 1)`` distinct
        :class:`SweepPoint` objects.

    Raises:
        ValueError: If an axis is empty, duplicated, or out of range.
    """
    _validate_axes(temperatures, top_ps)
    epoch_axis: Sequence[int | None] = list(epochs) if epochs else [None]
    if epochs is not None:
        if not epochs:
            raise ValueError("epochs must not be empty; pass None for a single checkpoint.")
        if len(set(epochs)) != len(epochs):
            raise ValueError(f"duplicate epochs in axis: {list(epochs)}")

    points: list[SweepPoint] = []
    for epoch in epoch_axis:
        checkpoint = None
        if checkpoints is not None and epoch is not None:
            checkpoint = checkpoints.get(epoch)
        for top_p in top_ps:
            for temperature in temperatures:
                points.append(
                    SweepPoint(
                        temperature=float(temperature),
                        top_p=float(top_p),
                        epoch=epoch,
                        checkpoint=checkpoint,
                    )
                )
    return points


#: The paper's temperature axis: 0.1 -> 2.0 in increments of 0.1 (20 values).
PAPER_TEMPERATURES: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(1, 21))

#: The paper's two nucleus-mass values.
PAPER_TOP_PS: tuple[float, ...] = (0.75, 0.95)

#: The paper's fine-tuning-epoch axis: 1 -> 15.
PAPER_EPOCHS: tuple[int, ...] = tuple(range(1, 16))

#: Polymers generated per configuration in the paper.
PAPER_SAMPLES_PER_CONFIG: int = 10_000

#: Configurations in the paper's full sweep: 3 model sizes x 15 epochs x 20
#: temperatures x 2 top_p.
PAPER_CONFIGURATIONS: int = 3 * len(PAPER_EPOCHS) * len(PAPER_TEMPERATURES) * len(PAPER_TOP_PS)

#: The paper's ``(temperature, top_p)`` plane for ONE checkpoint: 40 points.
#: The epoch axis is deliberately not baked in here -- it multiplies the cost by
#: 15 and requires 15 saved checkpoints, which a reproduction may not have.
PAPER_GRID: tuple[SweepPoint, ...] = tuple(
    sweep_grid(temperatures=PAPER_TEMPERATURES, top_ps=PAPER_TOP_PS)
)

#: Affordable temperature subset: the cold / paper-optimum / hot regimes.
#: 0.7 and 0.9 bracket the small model's reported optimum (0.9), 1.1 is the
#: medium/large optimum, and 1.5 probes the high-diversity, high-invalidity end.
#: # [AMBIGUITY] The paper defines no reduced grid; this subset is ours, chosen
#: # to straddle the three reported optima without leaving the paper's axes.
DEFAULT_TEMPERATURES: tuple[float, ...] = (0.7, 0.9, 1.1, 1.5)

#: Both of the paper's top_p values -- the axis is only two wide, and dropping
#: one would delete the validity-versus-diversity contrast the sweep is for.
DEFAULT_TOP_PS: tuple[float, ...] = PAPER_TOP_PS

#: A documented, affordable subset of :data:`PAPER_GRID`.
#:
#: COST RATIO. ``PAPER_GRID`` is 40 configurations against ``DEFAULT_GRID``'s 8
#: -- 5x. Against the paper's *full* sweep the ratio is far larger: 1,800
#: configurations x 10,000 samples = 18,000,000 generated polymers, versus
#: 8 x 200 = 1,600 here, a ratio of 11,250x. On the hardware this was developed
#: on (RTX 4080 Laptop, 12 GB), polyT5-small generates roughly 700 candidates in
#: well under a minute, so the full paper sweep is measured in GPU-weeks while
#: DEFAULT_GRID finishes in minutes. Always print the projected cost before
#: launching anything larger; ``scripts/sampling_sweep.py`` does exactly that
#: and refuses grids above ``--max-configs`` without ``--force``.
DEFAULT_GRID: tuple[SweepPoint, ...] = tuple(
    sweep_grid(temperatures=DEFAULT_TEMPERATURES, top_ps=DEFAULT_TOP_PS)
)


# ---------------------------------------------------------------------------
# one point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepResult:
    """Everything one sweep configuration produced.

    Attributes:
        point: The configuration that produced these numbers.
        n_requested: Candidates asked for (``n_samples``).
        counts: :meth:`polyt5.evaluation.FilterCounts.to_dict` output -- the
            nested SV/TSD/DD/PV counts and their rates over ``n_input``.
        sr_rate: SELFIES Reproducibility over all generated strings.
        duplicate_rate: Fraction of SV survivors repeating a polymer already
            seen earlier in this batch, i.e. ``1 - n_unique / n_sv``. This is
            the "increasing duplication" half of the paper's trade-off.
        mean_length: Mean number of SELFIES tokens per generated candidate,
            over all generated strings including the ones that fail screening.
            # [AMBIGUITY] the paper reports no length statistic; token count is
            # used because it is the unit the decoder's budget is spent in.
        tp_rate: The paper's TP metric, or ``None`` when no predictor was
            injected or nothing survived screening.
        property_mean: Mean predicted property over screened candidates, or
            ``None``.
        property_mae: Mean absolute error of the predicted property against the
            value each candidate was CONDITIONED on, or ``None``. This measures
            prompt obedience, which is what a conditional generator is for.
        seconds: Wall-clock seconds spent on this point, decoding plus
            screening plus prediction.
    """

    point: SweepPoint
    n_requested: int
    counts: dict[str, Any]
    sr_rate: float
    duplicate_rate: float
    mean_length: float
    tp_rate: float | None
    property_mean: float | None
    property_mae: float | None
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, JSON-serialisable row -- one sweep point, one record."""
        row: dict[str, Any] = dict(self.point.to_dict())
        row["n_requested"] = self.n_requested
        row.update(self.counts)
        row["sr_rate"] = self.sr_rate
        row["duplicate_rate"] = self.duplicate_rate
        row["mean_length"] = self.mean_length
        row["tp_rate"] = self.tp_rate
        row["property_mean"] = self.property_mean
        row["property_mae"] = self.property_mae
        row["seconds"] = self.seconds
        return row

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> SweepResult:
        """Rebuild a result from a :meth:`to_dict` row.

        This is what makes ``sweep.jsonl`` a genuine checkpoint rather than a
        log: a resumed run reloads its finished configurations and tabulates
        them alongside the new ones.

        Args:
            row: A flat row previously produced by :meth:`to_dict`.

        Returns:
            The reconstructed :class:`SweepResult`.

        Raises:
            KeyError: If a required key is absent, which means the row was not
                written by :meth:`to_dict` and must not be silently patched up.
        """
        point = SweepPoint(
            temperature=float(row["temperature"]),
            top_p=float(row["top_p"]),
            epoch=row.get("epoch"),
            checkpoint=row.get("checkpoint"),
        )
        counts = {key: row[key] for key in COUNT_KEYS}
        return cls(
            point=point,
            n_requested=int(row["n_requested"]),
            counts=counts,
            sr_rate=float(row["sr_rate"]),
            duplicate_rate=float(row["duplicate_rate"]),
            mean_length=float(row["mean_length"]),
            tp_rate=_finite(row.get("tp_rate")),
            property_mean=_finite(row.get("property_mean")),
            property_mae=_finite(row.get("property_mae")),
            seconds=float(row["seconds"]),
        )


#: The keys of :meth:`polyt5.evaluation.FilterCounts.to_dict`, in cascade order.
COUNT_KEYS: tuple[str, ...] = (
    "n_input",
    "n_sv",
    "n_tsd",
    "n_dd",
    "n_pv",
    "sv_rate",
    "tsd_rate",
    "dd_rate",
    "pv_rate",
)

#: Column order used by the dataframe, the CSV and the markdown table.
SWEEP_COLUMNS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "epoch",
    "n_requested",
    "n_input",
    "n_sv",
    "n_tsd",
    "n_dd",
    "n_pv",
    "sv_rate",
    "tsd_rate",
    "dd_rate",
    "pv_rate",
    "sr_rate",
    "duplicate_rate",
    "mean_length",
    "tp_rate",
    "property_mean",
    "property_mae",
    "seconds",
)


def assign_targets(targets: Sequence[float], n_samples: int) -> list[float]:
    """Spread ``n_samples`` prompts round-robin over the requested targets.

    Round-robin rather than blocked, so that a run interrupted part-way through
    a point still holds a balanced sample of every target.

    Args:
        targets: Property values to condition on, e.g. ``[300.0, 500.0]``.
        n_samples: How many candidates to generate in total.

    Returns:
        A list of ``n_samples`` target values, cycling through ``targets``.

    Raises:
        ValueError: If ``targets`` is empty or ``n_samples < 1``.

    Note:
        # [AMBIGUITY] The paper generates 10,000 polymers per configuration but
        # never says across how many conditioning targets, nor how they are
        # allocated. An even round-robin split is used here.
    """
    if not targets:
        raise ValueError("targets must not be empty.")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
    return [float(targets[i % len(targets)]) for i in range(n_samples)]


def generate_candidates(
    model: Any,
    tokenizer: _Tokenizer,
    *,
    point: SweepPoint,
    targets: Sequence[float],
    n_samples: int,
    device: Any,
    batch_size: int = 64,
    seed: int = 0,
    max_length: int = 200,
    decimals: int = 1,
    eval_mode: bool = True,
) -> list[str]:
    """Sample ``n_samples`` PSELFIES candidates at one sweep point.

    The prompt is the BARE numeric string, per the paper's SI -- ``"500.0"``,
    not ``"Tg: 500.0"`` -- built with
    :func:`polyt5.data.prepare.format_property_value`, which is the exact
    formatting the fine-tuning data used.

    Determinism: batch ``b`` decodes with seed ``seed + b``, so the result
    depends on ``seed`` and ``batch_size`` but not on machine load, and two
    calls with the same arguments return identical strings. Sampling uses a
    private :class:`torch.Generator` and never touches the global RNG.
    # [AMBIGUITY] The paper reports no seeds and no batching scheme, so the
    # per-batch seed offset is our choice; it exists so that two batches of the
    # same prompt do not decode to identical samples.

    Args:
        model: A :class:`polyt5.model.PolyT5ForConditionalGeneration`, or any
            module honouring the same ``encode`` / ``decode_step`` contract.
        tokenizer: The polyT5 tokenizer.
        point: The temperature / ``top_p`` configuration to sample at.
        targets: Property values to condition on.
        n_samples: Total candidates to generate.
        device: Torch device (or device string) to decode on.
        batch_size: Prompts per decoding batch.
        seed: Base RNG seed.
        max_length: Maximum tokens to generate (the paper uses 200).
        decimals: Decimal places in the prompt (the paper's SI shows one).
        eval_mode: Put the model in eval mode for the duration and restore the
            previous mode afterwards. Left on by default because dropout would
            silently destroy the determinism this sweep depends on; turn it off
            only if you deliberately want a stochastic-forward rollout.

    Returns:
        ``n_samples`` decoded PSELFIES strings, in prompt order.

    Raises:
        ValueError: If ``batch_size < 1``, ``n_samples < 1``, or ``targets`` is
            empty.
    """
    import torch

    from polyt5.generation import GenerationConfig, generate

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    sample_targets = assign_targets(targets, n_samples)
    prompts = [format_property_value(value, decimals=decimals) for value in sample_targets]
    torch_device = torch.device(device) if not isinstance(device, torch.device) else device

    was_training = bool(getattr(model, "training", False))
    if eval_mode and hasattr(model, "eval"):
        model.eval()

    candidates: list[str] = []
    try:
        for index, start in enumerate(range(0, len(prompts), batch_size)):
            chunk = prompts[start : start + batch_size]
            encoded = tokenizer.batch_encode(
                chunk, add_eos=True, max_length=max_length, padding=True, truncation=True
            )
            input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=torch_device)
            attention_mask = torch.tensor(
                encoded["attention_mask"], dtype=torch.long, device=torch_device
            )
            config = GenerationConfig(
                max_length=max_length,
                do_sample=True,
                temperature=float(point.temperature),
                top_p=float(point.top_p),
                eos_token_id=tokenizer.eos_id,
                pad_token_id=tokenizer.pad_id,
                decoder_start_token_id=tokenizer.decoder_start_token_id,
                seed=seed + index,
            )
            output = generate(model, input_ids, attention_mask, config=config)
            candidates.extend(
                tokenizer.batch_decode(output.sequences.tolist(), skip_special_tokens=True)
            )
    finally:
        if eval_mode and was_training and hasattr(model, "train"):
            model.train()
    return candidates


def _mean_selfies_length(candidates: Sequence[str]) -> float:
    """Mean number of SELFIES tokens per candidate, ``0.0`` for an empty batch."""
    if not candidates:
        return 0.0
    total = sum(len(_SELFIES_TOKEN_RE.findall(text)) for text in candidates)
    return total / len(candidates)


def _finite(value: Any) -> float | None:
    """Coerce ``value`` to a finite float, or ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return as_float if math.isfinite(as_float) else None


@dataclass(frozen=True)
class ScreenedBatch:
    """Generated candidates plus their per-candidate cascade outcomes.

    Produced by :func:`screen_candidates`, factored out of
    :func:`run_sweep_point` so a caller that needs PER-CANDIDATE detail --
    not just the aggregate a :class:`SweepResult` reports -- can reuse the
    exact same generation-and-screening pipeline instead of duplicating it.
    ``scripts/compare_arms.py`` is the motivating caller: scoring a
    composite or constraint reward needs the raw candidates, each one's own
    conditioning target, and (for a confidence-weighted term) a per-candidate
    ensemble std that no aggregate ``SweepResult`` can reconstruct -- reward
    scoring for those is done by feeding these fields straight into a
    :class:`polyt5.rewards.ArmReward`, not by re-deriving it from counts.

    Attributes:
        point: The configuration that produced these candidates.
        candidates: Raw generated PSELFIES strings, in prompt order.
        sample_targets: The conditioning target each candidate was prompted
            with, positionally aligned with ``candidates``.
        records: Per-candidate cascade outcomes from
            :func:`~polyt5.evaluation.filters.apply_filter_cascade`, same
            order as ``candidates``.
        counts: The nested SV/TSD/DD/PV counts for this batch.
        sr_rate: SELFIES Reproducibility over all generated strings.
        duplicate_rate: Fraction of SV survivors repeating a polymer already
            seen earlier in this batch.
        mean_length: Mean SELFIES-token length over all generated strings,
            including ones that fail screening.
    """

    point: SweepPoint
    candidates: list[str]
    sample_targets: list[float]
    records: list[CandidateRecord]
    counts: FilterCounts
    sr_rate: float
    duplicate_rate: float
    mean_length: float

    @property
    def screened_psmiles(self) -> list[str]:
        """Canonical PSMILES of the PV survivors, in input order."""
        return [
            record.canonical_psmiles
            for record in self.records
            if record.passed_pv and record.canonical_psmiles is not None
        ]

    @property
    def screened_targets(self) -> list[float]:
        """Conditioning target of each PV survivor, aligned with ``screened_psmiles``."""
        return [
            self.sample_targets[index]
            for index, record in enumerate(self.records)
            if record.passed_pv and record.canonical_psmiles is not None
        ]


def screen_candidates(
    model: Any,
    tokenizer: _Tokenizer,
    *,
    point: SweepPoint,
    targets: Sequence[float],
    n_samples: int,
    training_index: NoveltyIndex | None,
    device: Any,
    batch_size: int = 64,
    seed: int = 0,
    max_length: int = 200,
    compute_sa: bool = False,
) -> ScreenedBatch:
    """Generate and screen one configuration -- the shared core of :func:`run_sweep_point`.

    Args:
        model: The generation model, in or out of eval mode (put in eval mode
            for the duration by :func:`generate_candidates`) and on any
            device -- moved onto ``device`` in place before generation
            starts if it exposes ``.to`` (every real ``nn.Module`` does).
        tokenizer: The polyT5 tokenizer.
        point: The temperature / ``top_p`` configuration to sample at.
        targets: Property values to condition on, cycled round-robin.
        n_samples: Candidates to generate.
        training_index: Known training polymers for the TSD stage; ``None``
            makes TSD a no-op.
        device: Torch device (or device string) to decode on. ``model`` is
            moved here first, so a checkpoint loaded with
            ``map_location="cpu"`` -- the normal way to load one -- still
            ends up matched with the inputs this function builds on
            ``device``.
        batch_size: Prompts per decoding batch.
        seed: Base RNG seed; the result is fully determined by it.
        max_length: Maximum tokens to generate.
        compute_sa: Compute synthetic accessibility per candidate. Off by
            default (SA is the most expensive cascade step); the reward arms
            in :mod:`polyt5.rewards` compute their own SA when they need it,
            so most callers of this function do not need it precomputed here.

    Returns:
        A :class:`ScreenedBatch`. This function never raises on model output.
    """
    import torch

    # This function OWNS the device contract: it is what takes ``device`` as
    # an explicit parameter and builds ``generate_candidates``'s inputs on
    # it, so a caller that hands in a model still sitting on whatever device
    # it was loaded/constructed on (checkpoints load to CPU by convention;
    # see ``scripts/compare_arms.py``'s loaders) must not have to remember to
    # move it itself. ``model.to`` is in place for an ``nn.Module``, so this
    # mutates the caller's model rather than returning a new one -- standard
    # torch practice, and the only way a plain fake-model test double (never
    # constructed via a loader) still ends up on the right device too.
    torch_device = torch.device(device) if not isinstance(device, torch.device) else device
    if hasattr(model, "to"):
        model.to(torch_device)

    sample_targets = assign_targets(targets, n_samples)
    candidates = generate_candidates(
        model,
        tokenizer,
        point=point,
        targets=targets,
        n_samples=n_samples,
        device=device,
        batch_size=batch_size,
        seed=seed,
        max_length=max_length,
    )

    records, counts = apply_filter_cascade(
        candidates, training_index=training_index, expected_termini=2, compute_sa=compute_sa
    )

    n_reproducible = sum(1 for record in records if record.reproducible)
    sr_rate = (n_reproducible / counts.n_input) if counts.n_input else 0.0

    valid_canonical = [
        record.canonical_psmiles
        for record in records
        if record.passed_sv and record.canonical_psmiles is not None
    ]
    n_unique = len(dict.fromkeys(valid_canonical))
    duplicate_rate = (1.0 - n_unique / counts.n_sv) if counts.n_sv else 0.0

    return ScreenedBatch(
        point=point,
        candidates=candidates,
        sample_targets=sample_targets,
        records=records,
        counts=counts,
        sr_rate=sr_rate,
        duplicate_rate=duplicate_rate,
        mean_length=_mean_selfies_length(candidates),
    )


def property_columns(
    screened: Sequence[str],
    screened_targets: Sequence[float],
    predictor: Callable[[Sequence[str]], Sequence[float]] | None,
    target_property: float | None,
    tolerance: float,
) -> tuple[float | None, float | None, float | None]:
    """Run the injected predictor and derive ``(tp_rate, mean, mae)``.

    Args:
        screened: Canonical PSMILES that survived the whole cascade.
        screened_targets: The property value each screened candidate was
            conditioned on, aligned with ``screened``.
        predictor: Injected ``Callable[[Sequence[str]], Sequence[float]]``, or
            ``None``.
        target_property: Centre of the fixed TP window. When ``None`` the TP
            window is centred on each candidate's OWN conditioning target,
            which generalises the paper's single-target screen to a multi-target
            sweep.
            # [AMBIGUITY] Figure S10 only ever screens against a single target
            # (500 +- 50 K); the paper says nothing about a multi-target sweep.
        tolerance: Half-width of the TP window (the paper uses 50 K).

    Returns:
        ``(tp_rate, property_mean, property_mae)``, each ``None`` when it could
        not be measured. Nothing is ever estimated: no predictor, a predictor
        that raises, and a batch with no survivors all yield ``None`` rather
        than ``0.0``, because "not measured" is not the same as "measured zero".
    """
    if predictor is None or not screened:
        return (None, None, None)

    try:
        raw_values = list(predictor(list(screened)))
    except Exception:
        # An injected predictor is caller code; a failure there must not turn a
        # sweep into a crash, and must not be reported as a number.
        return (None, None, None)

    pairs: list[tuple[float, float]] = []
    for value, own_target in zip(raw_values, screened_targets, strict=False):
        finite = _finite(value)
        if finite is not None:
            pairs.append((finite, float(own_target)))
    if not pairs:
        return (None, None, None)

    values = [value for value, _ in pairs]
    property_mean = sum(values) / len(values)
    property_mae = sum(abs(value - own) for value, own in pairs) / len(pairs)

    if target_property is not None:
        tp_rate = target_property_rate(values, float(target_property), tolerance)
    else:
        hits = sum(1 for value, own in pairs if abs(value - own) <= tolerance)
        tp_rate = hits / len(pairs)
    return (tp_rate, property_mean, property_mae)


def run_sweep_point(
    model: Any,
    tokenizer: _Tokenizer,
    *,
    point: SweepPoint,
    targets: Sequence[float],
    n_samples: int,
    training_index: NoveltyIndex | None,
    property_predictor: Callable[[Sequence[str]], Sequence[float]] | None = None,
    target_property: float | None = None,
    tolerance: float = 50.0,
    device: Any,
    batch_size: int = 64,
    seed: int = 0,
    max_length: int = 200,
) -> SweepResult:
    """Generate, screen and score one sweep configuration.

    The pipeline is: sample ``n_samples`` candidates conditioned on ``targets``
    -> decode -> run the paper's nested SV/TSD/DD/PV cascade -> compute SR,
    the duplicate rate and the mean length -> and, ONLY if a predictor was
    injected, the TP rate and the property MAE.

    Synthetic accessibility is deliberately not computed: it is the most
    expensive step in the cascade and does not participate in any sweep
    objective, so paying for it once per configuration would dominate the run.
    # [AMBIGUITY] the paper does not say whether its sweep scored SA.

    Args:
        model: The generation model, in or out of eval mode (it is put in eval
            mode for the duration and restored afterwards).
        tokenizer: The polyT5 tokenizer.
        point: The configuration to evaluate.
        targets: Property values to condition on, cycled round-robin.
        n_samples: Candidates to generate (the paper uses 10,000).
        training_index: Known training polymers for the TSD stage; ``None``
            makes TSD a no-op.
        property_predictor: Injected ``Callable[[Sequence[str]], Sequence[float]]``
            mapping screened PSMILES to predicted property values. ``None``
            leaves every property column ``None``.
        target_property: Fixed TP window centre. ``None`` centres the window on
            each candidate's own conditioning target.
        tolerance: TP window half-width; the paper uses 50 K.
        device: Torch device (or device string) to decode on.
        batch_size: Prompts per decoding batch.
        seed: Base RNG seed; the result is fully determined by it.
        max_length: Maximum tokens to generate.

    Returns:
        A :class:`SweepResult`. This function never raises on model output.
    """
    started = time.perf_counter()
    batch = screen_candidates(
        model,
        tokenizer,
        point=point,
        targets=targets,
        n_samples=n_samples,
        training_index=training_index,
        device=device,
        batch_size=batch_size,
        seed=seed,
        max_length=max_length,
        compute_sa=False,
    )

    tp_rate, property_mean, property_mae = property_columns(
        batch.screened_psmiles, batch.screened_targets, property_predictor, target_property,
        tolerance,
    )

    return SweepResult(
        point=point,
        n_requested=n_samples,
        counts=batch.counts.to_dict(),
        sr_rate=batch.sr_rate,
        duplicate_rate=batch.duplicate_rate,
        mean_length=batch.mean_length,
        tp_rate=tp_rate,
        property_mean=property_mean,
        property_mae=property_mae,
        seconds=time.perf_counter() - started,
    )


# ---------------------------------------------------------------------------
# choosing a winner
# ---------------------------------------------------------------------------


def _objective_pv_rate(result: SweepResult) -> float | None:
    """Fraction of generated strings surviving the whole screen."""
    return _finite(result.counts.get("pv_rate"))


def _objective_n_pv(result: SweepResult) -> float | None:
    """Raw count of fully screened candidates -- the paper's own quantity."""
    return _finite(result.counts.get("n_pv"))


def _objective_sv_rate(result: SweepResult) -> float | None:
    """Fraction of generated strings RDKit accepts."""
    return _finite(result.counts.get("sv_rate"))


def _objective_sr_rate(result: SweepResult) -> float | None:
    """SELFIES round-trip reproducibility."""
    return _finite(result.sr_rate)


def _objective_tp_rate(result: SweepResult) -> float | None:
    """Fraction of screened candidates inside the target property window."""
    return _finite(result.tp_rate)


def _objective_composite(result: SweepResult) -> float | None:
    """Screening yield discounted by how repetitive the batch was.

    ``composite = pv_rate * (1 - duplicate_rate)``: the fraction of generated
    strings that are both fully valid and not a rewrite of something already
    produced in the same batch. It exists because ``pv_rate`` alone is maximised
    by cold, repetitive sampling -- the exact behaviour the paper describes as
    "reduced invalid outputs but increasing duplication", and the exact way a
    tuned baseline can look good while generating almost nothing new.

    # [AMBIGUITY] This objective is OURS, not the paper's. The paper selects on
    # the number of valid candidates alone; the product form here is the
    # simplest way to price duplication into that selection, and it is offered
    # alongside the paper's own criterion rather than replacing it.
    """
    pv_rate = _finite(result.counts.get("pv_rate"))
    duplicate_rate = _finite(result.duplicate_rate)
    if pv_rate is None or duplicate_rate is None:
        return None
    return pv_rate * (1.0 - duplicate_rate)


#: Selectable objectives for :func:`select_best`.
OBJECTIVES: dict[str, Callable[[SweepResult], float | None]] = {
    "pv_rate": _objective_pv_rate,
    "n_pv": _objective_n_pv,
    "sv_rate": _objective_sv_rate,
    "sr_rate": _objective_sr_rate,
    "tp_rate": _objective_tp_rate,
    "composite": _objective_composite,
}


def select_best(results: Sequence[SweepResult], *, objective: str = "pv_rate") -> SweepResult:
    """Pick the winning configuration under an explicit objective.

    THE OBJECTIVE IS THE WHOLE ARGUMENT. The paper optimises for the *number of
    valid candidates* -- its reported optima are the peaks of PV-count heatmaps
    -- and optimising PV alone favours low temperature and therefore high
    duplication. Since a tuned supervised baseline that merely repeats itself
    would flatter Arm B and mislead the eventual RL comparison, the objective is
    switchable and never hard-coded. Available objectives:

    ==============  =========================================================
    ``pv_rate``     Fraction of generated strings surviving the whole screen
                    (the paper's quantity, normalised). Default.
    ``n_pv``        Raw count of screened candidates -- literally what the
                    paper's heatmaps show. Ranks identically to ``pv_rate``
                    when every configuration requests the same ``n_samples``.
    ``sv_rate``     RDKit validity only.
    ``sr_rate``     SELFIES round-trip reproducibility.
    ``tp_rate``     Fraction of screened candidates hitting the property
                    window; requires an injected predictor.
    ``composite``   ``pv_rate * (1 - duplicate_rate)`` -- validity discounted
                    by duplication. See :func:`_objective_composite`.
    ==============  =========================================================

    Args:
        results: Sweep results to choose among.
        objective: Key of :data:`OBJECTIVES`.

    Returns:
        The result with the highest objective value.

    Raises:
        ValueError: If ``results`` is empty, ``objective`` is unknown, or no
            result has a measurable value for the requested objective (for
            instance ``tp_rate`` with no predictor injected -- reporting an
            arbitrary winner there would be fabricating the comparison).

    Note:
        TIE-BREAK: the FIRST result in the given order that attains the maximum
        wins; ties are never broken by temperature or by any other hidden
        preference. Selection therefore depends only on the caller's ordering,
        and is stable -- re-selecting from the same list returns the same
        object every time.
    """
    if not results:
        raise ValueError("select_best requires at least one result.")
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; choose from {sorted(OBJECTIVES)}.")

    score = OBJECTIVES[objective]
    best: SweepResult | None = None
    best_value: float | None = None
    for result in results:
        value = score(result)
        if value is None:
            continue
        if best_value is None or value > best_value:
            best, best_value = result, value
    if best is None:
        raise ValueError(
            f"no result has a measurable {objective!r} value "
            "(a tp_rate objective needs an injected property predictor)."
        )
    return best


# ---------------------------------------------------------------------------
# tabulation -- the numbers the paper never published
# ---------------------------------------------------------------------------


def _rows(results: Sequence[SweepResult]) -> list[dict[str, Any]]:
    """Flatten results into ordered row dicts using :data:`SWEEP_COLUMNS`."""
    rows: list[dict[str, Any]] = []
    for result in results:
        flat = result.to_dict()
        row = {column: flat.get(column) for column in SWEEP_COLUMNS}
        row["checkpoint"] = flat.get("checkpoint")
        rows.append(row)
    return rows


def sweep_to_dataframe(results: Sequence[SweepResult]) -> pd.DataFrame:
    """Tabulate sweep results as a pandas DataFrame.

    Args:
        results: Results to tabulate, in any order (row order is preserved).

    Returns:
        A DataFrame with one row per result and the columns of
        :data:`SWEEP_COLUMNS`, plus ``checkpoint``. Unmeasured property columns
        stay ``None``/``NaN`` rather than being filled in.
    """
    import pandas as pd

    columns = [*SWEEP_COLUMNS, "checkpoint"]
    return pd.DataFrame(_rows(results), columns=columns)


def _sorted_rows(rows: list[dict[str, Any]], sort_by: str | None) -> list[dict[str, Any]]:
    """Sort rows descending by ``sort_by``; unmeasured values sink to the end.

    Args:
        rows: Flat row dicts.
        sort_by: Column to sort by, or ``None`` to keep input order.

    Returns:
        The sorted rows.

    Raises:
        ValueError: If ``sort_by`` is not a known column.
    """
    if sort_by is None:
        return rows
    if sort_by not in SWEEP_COLUMNS:
        raise ValueError(f"unknown sort_by {sort_by!r}; choose from {list(SWEEP_COLUMNS)}.")

    def key(row: dict[str, Any]) -> tuple[int, float]:
        value = _finite(row.get(sort_by))
        return (0, -value) if value is not None else (1, 0.0)

    return sorted(rows, key=key)


def _format_cell(column: str, value: Any) -> str:
    """Render one cell, never inventing a value for a missing measurement."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    if column in {"temperature", "top_p"}:
        return f"{float(value):.2f}"
    if column == "seconds":
        return f"{float(value):.1f}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def sweep_to_markdown(results: Sequence[SweepResult], *, sort_by: str | None = "pv_rate") -> str:
    """Render the numeric sweep table the paper never published.

    Sahu et al. show their per-configuration counts only as unlabeled heatmaps,
    so no published table exists to diff this against; the point of emitting
    markdown is that the reproduction's own numbers are readable and citable.

    Args:
        results: Results to render.
        sort_by: Column to sort descending by (default ``"pv_rate"``), or
            ``None`` to keep input order. Rows with no value for the sort
            column are placed last.

    Returns:
        A GitHub-flavoured markdown table: a header naming every metric, an
        alignment row, then one row per result. Unmeasured cells read ``n/a``.

    Raises:
        ValueError: If ``sort_by`` names a column that does not exist.
    """
    columns = list(SWEEP_COLUMNS)
    rows = _sorted_rows(_rows(results), sort_by)

    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---:" for _ in columns) + "|"]
    for row in rows:
        lines.append(
            "| " + " | ".join(_format_cell(column, row.get(column)) for column in columns) + " |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# incremental persistence
# ---------------------------------------------------------------------------


def append_result_jsonl(path: str | Path, result: SweepResult) -> Path:
    """Append one result to a JSON Lines file, flushed immediately.

    A sweep is a long, interruptible job, so every point is durable the moment
    it finishes: the file is opened in append mode, one line is written, and
    the handle is flushed and closed before the next point starts. A run killed
    part-way therefore keeps every completed point.

    Args:
        path: Destination ``.jsonl`` file; parent directories are created.
        result: The result to append.

    Returns:
        The path written to.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.to_dict(), default=str) + "\n")
        handle.flush()
    return destination


def read_results_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read back rows written by :func:`append_result_jsonl`.

    Args:
        path: The ``.jsonl`` file. A missing file reads as empty.

    Returns:
        One dict per parseable line. Blank lines and a truncated final line --
        the signature of a process killed mid-write -- are skipped rather than
        raising, so an interrupted sweep is still analysable.
    """
    source = Path(path)
    if not source.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows
