"""Torch-side property predictor: the callable the evaluation layer is missing.

:func:`polyt5.evaluation.evaluate_generation` takes its property predictor as an
*injected* ``Callable[[Sequence[str]], Sequence[float]]`` so that the evaluation
package -- and therefore the future RL reward worker -- never imports torch.
Nothing in the repository supplied that callable, so the paper's **TP** metric
("proportion of candidates with predicted Tg within 500 +- 50 K") always came
back ``None``. :class:`PolyT5PropertyPredictor` is that callable.

The dependency arrow points one way only::

    polyt5.inference  ---->  polyt5.evaluation        (allowed, and used here)
    polyt5.evaluation  -/->  polyt5.inference         (never; keeps torch out)

Decoding follows the paper's property-prediction protocol exactly: the input is
the **bare PSELFIES with no task prefix**, decoding is **beam search of width
4**, and the output is a bare number with one decimal (``"236.0"``). Parsing
reuses :func:`polyt5.evaluation.parse_numeric_predictions`, so what this module
calls numeric is bit-for-bit what :func:`polyt5.evaluation.regression_report`
calls numeric.

Contract for a non-numeric decode (the important one):

* :meth:`PolyT5PropertyPredictor.predict` reports ``is_numeric=False`` and
  ``value=None`` -- an explicit "no answer".
* :meth:`PolyT5PropertyPredictor.__call__`, whose return type is ``float``, maps
  it to ``float("nan")``. It is deliberately **not** ``0.0``: a zero would read
  downstream as a genuine prediction of 0 K, would drag ``property_mean`` toward
  zero, and would be counted as an out-of-window miss by the TP metric.
  :func:`polyt5.evaluation.target_property_rate` documents that non-finite
  entries are dropped from *both* the numerator and the denominator, so NaN is
  the value that makes "the model produced no answer" disappear from TP instead
  of masquerading as a failure to hit the window.

Exception safety: this module is fed raw model output, so it never raises on it.
A decode that is not a number is a recorded failure; a string that is neither
PSELFIES nor PSMILES is a recorded failure; only genuine caller errors (a
missing checkpoint, a tokenizer mismatch, a nonsensical batch size) raise.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch

from polyt5.chemistry import psmiles_to_pselfies
from polyt5.evaluation import parse_numeric_predictions
from polyt5.generation import BeamSearchConfig, beam_search
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.tokenization import PolyT5Tokenizer
from polyt5.training import load_checkpoint
from polyt5.utils import select_device

__all__ = [
    "DEFAULT_MAX_SOURCE_LENGTH",
    "DEFAULT_TOLERANCES",
    "NON_NUMERIC_VALUE",
    "CacheInfo",
    "CachedPredictor",
    "ConditioningReport",
    "EnsemblePropertyPredictor",
    "PolyT5PropertyPredictor",
    "PredictionResult",
    "conditioning_report",
    "looks_like_pselfies",
]

#: What :meth:`PolyT5PropertyPredictor.__call__` returns when no number came
#: back. See the module docstring for why this is NaN and not 0.0.
NON_NUMERIC_VALUE: float = float("nan")

#: Paper maximum input length in tokens.
DEFAULT_MAX_SOURCE_LENGTH: int = 200

#: Default acceptance windows, in K, for :func:`conditioning_report`.
#: # [AMBIGUITY] The paper reports a single fixed 500 +- 50 K screen and never
#: measures per-candidate conditioning error, so it names no tolerance ladder;
#: 25/50/100 K brackets the observed error scale.
DEFAULT_TOLERANCES: tuple[float, ...] = (25.0, 50.0, 100.0)

# A PSELFIES string is a concatenation of bracket tokens and nothing else
# ("[At][C][C][O][At]"). A PSMILES string carries bare atoms between the
# brackets ("[At]CCO[At]"), so this cheap structural test separates the two
# notations without invoking RDKit.
_PSELFIES_RE = re.compile(r"^(?:\[[^\[\]]+\])+$")


@dataclass(frozen=True)
class PredictionResult:
    """One model prediction, with the decode that produced it kept alongside.

    Attributes:
        source: The string the caller supplied, verbatim.
        decoded: What the model actually emitted, after the tokenizer dropped
            special tokens. Retained even when it is not a number, because the
            failure text is the only evidence of *how* the model failed.
        value: The parsed property value, or ``None`` when the decode was not a
            number.
        is_numeric: Whether ``value`` is present. Equivalent to
            ``value is not None``, spelled out so a caller reading a serialised
            record does not have to infer it.
    """

    source: str
    decoded: str
    value: float | None
    is_numeric: bool


class CacheInfo(NamedTuple):
    """Cache statistics, mirroring :func:`functools.lru_cache`'s field names."""

    hits: int
    misses: int
    maxsize: int | None
    currsize: int


@dataclass(frozen=True)
class ConditioningReport:
    """How closely conditional generation hit the target it was *asked* for.

    The paper's TP metric answers "what fraction of candidates land in one fixed
    window?" -- appropriate when every candidate was conditioned on the same
    value, as in its 500 +- 50 K screen. It answers the wrong question for a
    generations file whose rows carry different conditioning targets: measuring
    such a file against a fixed 500 K window scores candidates on a request
    nobody made of them.

    This report measures the metric that actually matters there, and that the
    RLVR reward will optimise: each candidate's predicted property against *its
    own* conditioning target.

    Attributes:
        n_candidates: Pairs offered, i.e. the length of the input sequences.
        n_scored: Pairs where BOTH a finite prediction and a finite target were
            available. Every statistic below is computed over exactly these.
        n_missing_target: Pairs dropped because the row carried no usable
            conditioning target. Counted, never defaulted to zero -- a missing
            target is not a target of 0 K.
        n_non_numeric_prediction: Pairs dropped because the predictor returned
            no finite value (see :data:`NON_NUMERIC_VALUE`).
        mae: Mean absolute error in the property's units, or ``None`` when
            nothing could be scored.
        median_abs_error: Median absolute error -- robust to the long tail that
            a handful of wild candidates puts on the mean.
        rmse: Root mean squared error.
        mean_signed_bias: Mean of ``predicted - target``. Reported separately
            from ``mae`` because the two describe different failures: a model
            uniformly 40 K low has ``mae == 40`` and ``bias == -40``, while one
            scattering +-40 K at random has ``mae == 40`` and ``bias ~= 0``.
            Negative means the candidates' predicted property sits BELOW what
            was asked for.
        within_tolerance: ``{"25": 0.345, ...}`` -- the fraction of scored
            pairs with ``|predicted - target| <= tolerance``, keyed by the
            tolerance formatted with ``%g``.
        tolerances: The tolerance ladder actually used, ascending.
        target_min: Smallest conditioning target among the scored pairs, or
            ``None``. Together with ``target_max`` this is what shows a reader
            whether the file was conditioned on one value or on a whole range.
        target_max: Largest conditioning target among the scored pairs.
        target_mean: Mean conditioning target among the scored pairs.
    """

    n_candidates: int
    n_scored: int
    n_missing_target: int
    n_non_numeric_prediction: int
    mae: float | None
    median_abs_error: float | None
    rmse: float | None
    mean_signed_bias: float | None
    within_tolerance: dict[str, float]
    tolerances: list[float]
    target_min: float | None
    target_max: float | None
    target_mean: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "n_candidates": self.n_candidates,
            "n_scored": self.n_scored,
            "n_missing_target": self.n_missing_target,
            "n_non_numeric_prediction": self.n_non_numeric_prediction,
            "mae": self.mae,
            "median_abs_error": self.median_abs_error,
            "rmse": self.rmse,
            "mean_signed_bias": self.mean_signed_bias,
            "within_tolerance": dict(self.within_tolerance),
            "tolerances": list(self.tolerances),
            "target_min": self.target_min,
            "target_max": self.target_max,
            "target_mean": self.target_mean,
        }


def conditioning_report(
    predicted: Sequence[Any],
    targets: Sequence[Any],
    *,
    tolerances: Sequence[float] = DEFAULT_TOLERANCES,
) -> ConditioningReport:
    """Score predictions against their own per-candidate conditioning targets.

    Pure arithmetic: no torch, no model, no I/O. It lives beside the predictor
    only because the predictor is what produces its left-hand argument.

    Args:
        predicted: Predicted property values, typically straight out of
            :meth:`PolyT5PropertyPredictor.__call__`, so non-finite entries mean
            "the predictor had no answer".
        targets: The conditioning target each candidate was generated for,
            positionally aligned with ``predicted``. ``None`` or a non-finite
            entry means the row carried no usable target.
        tolerances: Acceptance half-widths to report hit rates for. Duplicates
            are collapsed and the ladder is sorted ascending. Non-positive or
            non-finite entries are ignored.

    Returns:
        A :class:`ConditioningReport`. Every statistic is ``None`` when nothing
        could be scored -- never ``0.0``, which would read as a perfect model.

    Raises:
        ValueError: If the two sequences have different lengths. That is a
            caller bug -- a misalignment silently scores each candidate against
            somebody else's target -- so it is not swallowed.
    """
    if len(predicted) != len(targets):
        raise ValueError(
            f"predicted and targets must be the same length, got {len(predicted)} "
            f"and {len(targets)}"
        )

    ladder = sorted({
        float(tolerance)
        for tolerance in tolerances
        if _is_finite(tolerance) and float(tolerance) > 0
    })

    errors: list[float] = []
    scored_targets: list[float] = []
    n_missing_target = 0
    n_non_numeric = 0
    for prediction, target in zip(predicted, targets, strict=True):
        if not _is_finite(prediction):
            n_non_numeric += 1
            continue
        if not _is_finite(target):
            n_missing_target += 1
            continue
        errors.append(float(prediction) - float(target))
        scored_targets.append(float(target))

    if not errors:
        return ConditioningReport(
            n_candidates=len(predicted),
            n_scored=0,
            n_missing_target=n_missing_target,
            n_non_numeric_prediction=n_non_numeric,
            mae=None,
            median_abs_error=None,
            rmse=None,
            mean_signed_bias=None,
            within_tolerance={},
            tolerances=ladder,
            target_min=None,
            target_max=None,
            target_mean=None,
        )

    absolute = [abs(error) for error in errors]
    return ConditioningReport(
        n_candidates=len(predicted),
        n_scored=len(errors),
        n_missing_target=n_missing_target,
        n_non_numeric_prediction=n_non_numeric,
        mae=float(statistics.fmean(absolute)),
        median_abs_error=float(statistics.median(absolute)),
        rmse=math.sqrt(statistics.fmean([error * error for error in errors])),
        mean_signed_bias=float(statistics.fmean(errors)),
        within_tolerance={
            f"{tolerance:g}": sum(1 for value in absolute if value <= tolerance) / len(absolute)
            for tolerance in ladder
        },
        tolerances=ladder,
        target_min=min(scored_targets),
        target_max=max(scored_targets),
        target_mean=float(statistics.fmean(scored_targets)),
    )


def looks_like_pselfies(text: str) -> bool:
    """Return whether ``text`` is written in SELFIES bracket-token notation.

    Args:
        text: A candidate polymer string.

    Returns:
        ``True`` for an all-bracket-token string such as ``"[At][C][C][At]"``,
        ``False`` for PSMILES such as ``"[At]CCO[At]"`` or ``"[*]CCO[*]"``.
    """
    return bool(_PSELFIES_RE.match(text.strip()))


class PolyT5PropertyPredictor:
    """Loads a fine-tuned property checkpoint and scores PSELFIES strings.

    The primary API is PSELFIES, matching the paper's fine-tuning format.
    PSMILES is accepted through :meth:`predict_psmiles`, and :meth:`__call__`
    auto-detects the notation because
    :func:`polyt5.evaluation.evaluate_generation` hands its predictor the
    *canonical PSMILES* of the screened candidates.

    Example:
        >>> predictor = PolyT5PropertyPredictor.from_checkpoint(  # doctest: +SKIP
        ...     "results/finetune_tg_prediction/checkpoints/best.pt",
        ...     tokenizer_path="artifacts/tokenizer/polyt5_vocab.json",
        ... )
        >>> report = evaluate_generation(  # doctest: +SKIP
        ...     candidates, target_property=500.0, property_predictor=predictor
        ... )
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PolyT5Tokenizer,
        *,
        device: str = "auto",
        batch_size: int = 64,
        num_beams: int = 4,
        max_target_length: int = 32,
        property_name: str | None = None,
        max_source_length: int = DEFAULT_MAX_SOURCE_LENGTH,
    ) -> None:
        """Wrap a loaded model for inference.

        Args:
            model: A :class:`~polyt5.model.PolyT5ForConditionalGeneration`, or
                any module honouring the same ``encode`` / ``decode_step``
                contract that :func:`polyt5.generation.beam_search` calls.
            tokenizer: The tokenizer the model was trained with. Supplying a
                different vocabulary silently corrupts every token id, which is
                why :meth:`from_checkpoint` verifies the hash.
            device: ``"auto"``, ``"cpu"``, ``"cuda"``, or an explicit device
                string such as ``"cuda:1"``.
            batch_size: How many candidates are decoded per beam-search call.
                Purely a throughput knob: beam search is independent per row, so
                the batching must not change any prediction.
            num_beams: Beam width. # [PAPER] 4 for property prediction.
            max_target_length: Cap on generated tokens. A one-decimal number is
                a handful of tokens; 32 leaves room for a model that rambles
                without paying for the paper's 200-token cap on every call.
                # [AMBIGUITY] the paper states the 200-token *input* limit but
                # never a decode cap for the numeric target.
            property_name: Optional label ("Tg") carried for bookkeeping. It is
                never prepended to the input -- the paper's format has no task
                prefix.
            max_source_length: Input truncation length. # [PAPER] 200. Clamped
                down to the model's ``n_positions`` when that is smaller.

        Raises:
            ValueError: If ``batch_size``, ``num_beams`` or
                ``max_target_length`` is less than 1.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if num_beams < 1:
            raise ValueError(f"num_beams must be >= 1, got {num_beams}")
        if max_target_length < 1:
            raise ValueError(f"max_target_length must be >= 1, got {max_target_length}")

        self.device = select_device("auto") if device == "auto" else str(device)
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.num_beams = int(num_beams)
        self.max_target_length = int(max_target_length)
        self.property_name = property_name

        n_positions = getattr(getattr(model, "config", None), "n_positions", None)
        self.max_source_length = (
            min(int(max_source_length), int(n_positions))
            if isinstance(n_positions, int) and n_positions > 0
            else int(max_source_length)
        )

        self.model = model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------ construction
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path | None = None,
        *,
        device: str = "auto",
        allow_unverified_tokenizer: bool = False,
        **kwargs: Any,
    ) -> PolyT5PropertyPredictor:
        """Rebuild a predictor from a fine-tuned checkpoint.

        Args:
            checkpoint_path: A ``.pt`` file written by
                :func:`polyt5.training.save_checkpoint`.
            tokenizer_path: Tokenizer artifact to load. When ``None`` the path
                recorded inside the checkpoint is used.
            device: Passed through to :meth:`__init__`.
            allow_unverified_tokenizer: Permit a checkpoint that recorded no
                ``tokenizer_sha256``. Off by default: a wrong vocabulary at
                inference time yields plausible-looking wrong *numbers* rather
                than a crash, so an unverifiable checkpoint is refused unless
                the caller says otherwise.
            **kwargs: Forwarded to :meth:`__init__` (``batch_size``,
                ``num_beams``, ``max_target_length``, ``property_name``,
                ``max_source_length``).

        Returns:
            A ready-to-use predictor whose model is in ``eval`` mode.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
            ValueError: If no tokenizer can be located, if the checkpoint's
                ``tokenizer_sha256`` does not match the tokenizer actually
                loaded, or if the checkpoint recorded no hash and
                ``allow_unverified_tokenizer`` is ``False``.
        """
        checkpoint_path = Path(checkpoint_path)
        payload = load_checkpoint(checkpoint_path, map_location="cpu")

        resolved_tokenizer = tokenizer_path or payload.get("tokenizer_path")
        if resolved_tokenizer is None:
            raise ValueError(
                f"{checkpoint_path} records no tokenizer_path and none was supplied. "
                "Pass tokenizer_path=... explicitly; guessing a vocabulary would "
                "silently remap every token id."
            )
        tokenizer = PolyT5Tokenizer.from_file(Path(resolved_tokenizer))

        recorded_sha = payload.get("tokenizer_sha256")
        if recorded_sha is None:
            if not allow_unverified_tokenizer:
                raise ValueError(
                    f"{checkpoint_path} recorded no tokenizer_sha256, so the vocabulary "
                    "cannot be verified. Re-save the checkpoint with its tokenizer hash, "
                    "or pass allow_unverified_tokenizer=True to accept the risk."
                )
        elif recorded_sha != tokenizer.sha256:
            raise ValueError(
                "tokenizer mismatch: the checkpoint was trained with vocabulary "
                f"{recorded_sha[:16]} but the supplied tokenizer is "
                f"{tokenizer.sha256[:16]}. Scoring across vocabularies would silently "
                f"corrupt every token id.\n  checkpoint: {checkpoint_path}\n"
                f"  tokenizer:  {resolved_tokenizer}"
            )

        model_config = PolyT5Config.from_dict(payload["model_config"])
        model = PolyT5ForConditionalGeneration(model_config)
        model.load_state_dict(payload["model_state"])
        return cls(model, tokenizer, device=device, **kwargs)

    # -------------------------------------------------------------- prediction
    def predict(self, pselfies: Sequence[str]) -> list[PredictionResult]:
        """Score PSELFIES strings with the paper's beam-search decode.

        Args:
            pselfies: Candidate polymers as PSELFIES. Entries that are not
                non-empty strings are reported as non-numeric without being sent
                to the model.

        Returns:
            One :class:`PredictionResult` per input, in input order.

        Note:
            Never raises on model output. A decode that is not a number becomes
            ``is_numeric=False``; it is not an error and not a zero.
        """
        results: list[PredictionResult | None] = [None] * len(pselfies)
        live: list[tuple[int, str]] = []
        for position, entry in enumerate(pselfies):
            if isinstance(entry, str) and entry.strip():
                live.append((position, entry))
            else:
                source = entry if isinstance(entry, str) else str(entry)
                results[position] = PredictionResult(
                    source=source, decoded="", value=None, is_numeric=False
                )

        for start in range(0, len(live), self.batch_size):
            chunk = live[start : start + self.batch_size]
            decoded = self._decode([text for _, text in chunk])
            values, _ = parse_numeric_predictions(decoded)
            for (position, text), text_out, value in zip(chunk, decoded, values, strict=True):
                results[position] = PredictionResult(
                    source=text,
                    decoded=text_out,
                    value=value,
                    is_numeric=value is not None,
                )

        return [result for result in results if result is not None]

    def predict_values(self, pselfies: Sequence[str]) -> list[float | None]:
        """Score PSELFIES and return values only.

        Args:
            pselfies: Candidate polymers as PSELFIES.

        Returns:
            One ``float`` or ``None`` per input, in input order. ``None`` marks
            a non-numeric decode -- use :meth:`__call__` when the caller needs a
            uniform ``float`` and can handle NaN.
        """
        return [result.value for result in self.predict(pselfies)]

    def predict_psmiles(self, psmiles: Sequence[str]) -> list[PredictionResult]:
        """Score PSMILES strings by converting them to PSELFIES first.

        Convenience wrapper around :meth:`predict`; the model itself only ever
        sees PSELFIES. Both ``[*]`` and ``[At]`` terminus notations are
        accepted, since :func:`polyt5.chemistry.psmiles_to_pselfies` normalises
        them.

        Args:
            psmiles: Candidate polymers as PSMILES.

        Returns:
            One :class:`PredictionResult` per input, in input order, whose
            ``source`` is the PSMILES the caller passed. An entry that cannot be
            converted is reported as non-numeric rather than raising.
        """
        converted: list[str | None] = [
            psmiles_to_pselfies(entry) if isinstance(entry, str) else None for entry in psmiles
        ]
        encodable = [(i, value) for i, value in enumerate(converted) if value]
        scored = self.predict([value for _, value in encodable])

        results: list[PredictionResult] = [
            PredictionResult(
                source=entry if isinstance(entry, str) else str(entry),
                decoded="",
                value=None,
                is_numeric=False,
            )
            for entry in psmiles
        ]
        for (position, _), result in zip(encodable, scored, strict=True):
            source = psmiles[position]
            results[position] = PredictionResult(
                source=source if isinstance(source, str) else str(source),
                decoded=result.decoded,
                value=result.value,
                is_numeric=result.is_numeric,
            )
        return results

    def __call__(self, candidates: Sequence[str]) -> list[float]:
        """Score candidates, returning one ``float`` each. **The injection point.**

        This is the ``Callable[[Sequence[str]], Sequence[float]]`` that
        :func:`polyt5.evaluation.evaluate_generation` expects for its
        ``property_predictor`` argument. Because that function hands its
        predictor the *canonical PSMILES* of the screened candidates while the
        model is trained on PSELFIES, the notation is auto-detected per entry:
        an all-bracket-token string is used as PSELFIES, anything else is
        converted from PSMILES.

        Args:
            candidates: Polymers as PSELFIES or PSMILES, in any mixture.

        Returns:
            One ``float`` per input, in input order.
            :data:`NON_NUMERIC_VALUE` (NaN) marks an entry that produced no
            number -- an unusable input, an unconvertible PSMILES, a
            non-numeric decode, or an inference failure. NaN is chosen so the
            entry drops out of both the numerator and the denominator of
            :func:`polyt5.evaluation.target_property_rate`; ``0.0`` would be
            read as a real prediction of 0 K.
        """
        as_pselfies: list[str | None] = []
        for entry in candidates:
            if not isinstance(entry, str) or not entry.strip():
                as_pselfies.append(None)
            elif looks_like_pselfies(entry):
                as_pselfies.append(entry)
            else:
                as_pselfies.append(psmiles_to_pselfies(entry))

        encodable = [(i, value) for i, value in enumerate(as_pselfies) if value]
        values: list[float] = [NON_NUMERIC_VALUE] * len(as_pselfies)
        if not encodable:
            return values

        try:
            scored = self.predict_values([value for _, value in encodable])
        except Exception:
            # The injection point must be total: an inference failure is "no
            # numbers", never an exception escaping into an evaluation run.
            return values

        for (position, _), value in zip(encodable, scored, strict=True):
            values[position] = float(value) if value is not None else NON_NUMERIC_VALUE
        return values

    # ----------------------------------------------------------------- internal
    @torch.no_grad()
    def _decode(self, texts: Sequence[str]) -> list[str]:
        """Run one beam-search batch and return the decoded strings.

        Args:
            texts: PSELFIES strings, already filtered to non-empty entries.

        Returns:
            The decoded model outputs, positionally aligned with ``texts``.
        """
        if not texts:
            return []
        self.model.eval()
        encoded = self.tokenizer.batch_encode(
            texts,
            add_eos=True,
            max_length=self.max_source_length,
            padding=True,
            truncation=True,
        )
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(
            encoded["attention_mask"], dtype=torch.long, device=self.device
        )
        output = beam_search(
            self.model,
            input_ids,
            attention_mask,
            config=BeamSearchConfig(
                num_beams=self.num_beams,  # [PAPER] width 4
                max_length=self.max_target_length,
                eos_token_id=self.tokenizer.eos_id,
                pad_token_id=self.tokenizer.pad_id,
                decoder_start_token_id=self.tokenizer.decoder_start_token_id,
            ),
        )
        return self.tokenizer.batch_decode(output.sequences.tolist(), skip_special_tokens=True)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(property={self.property_name!r}, device={self.device!r}, "
            f"num_beams={self.num_beams}, batch_size={self.batch_size})"
        )


class EnsemblePropertyPredictor:
    """Averages several predictors and reports their disagreement.

    The five-random-split protocol (:mod:`scripts.run_splits`) produces five
    independently trained Tg models. Kept rather than discarded, they give the
    RL phase two things for free:

    * an **uncertainty estimate** -- the spread across members. A policy
      optimised against a single predictor will find the regions where that one
      model is confidently wrong and farm them; member disagreement is the
      cheapest available signal that a candidate sits there.
    * a **held-out auditor** -- reserve one member, never use it in any reward,
      and score final candidates with it.

    Non-finite handling (identical rule to :class:`PolyT5PropertyPredictor`, and
    to :func:`polyt5.evaluation.target_property_rate`): a member that returns no
    number is **dropped from that candidate's mean and std**, not counted as a
    zero and not allowed to poison the aggregate. The candidate's value is
    :data:`NON_NUMERIC_VALUE` (NaN) only when *no* member produced a number.
    ``predict_with_uncertainty`` also returns the contributing-member count, so
    "all five agreed" is never confused with "only one answered".
    """

    def __init__(self, predictors: Sequence[PolyT5PropertyPredictor]) -> None:
        """Wrap the ensemble members.

        Args:
            predictors: Two or more predictors, or any callables returning one
                value per candidate. One member is allowed but pointless: its
                std is always 0.

        Raises:
            ValueError: If ``predictors`` is empty -- an ensemble of nothing has
                no mean to report.
        """
        members = list(predictors)
        if not members:
            raise ValueError("EnsemblePropertyPredictor needs at least one member")
        self.predictors = members

    @classmethod
    def from_checkpoints(
        cls,
        paths: Sequence[str | Path],
        tokenizer_path: str | Path | None = None,
        *,
        device: str = "auto",
        **kwargs: Any,
    ) -> EnsemblePropertyPredictor:
        """Load one member per checkpoint.

        Args:
            paths: Checkpoint files, e.g. the ``split_*/checkpoints/best.pt``
                written by ``scripts/run_splits.py``.
            tokenizer_path: Tokenizer artifact shared by every member. When
                ``None``, each checkpoint's own recorded path is used. Every
                member is hash-verified individually, so a checkpoint trained on
                a different vocabulary is refused here rather than quietly
                averaged in.
            device: Passed through to each member.
            **kwargs: Forwarded to :meth:`PolyT5PropertyPredictor.from_checkpoint`.

        Returns:
            The assembled ensemble.

        Raises:
            ValueError: If ``paths`` is empty, or any member fails its checks.
        """
        if not paths:
            raise ValueError("from_checkpoints needs at least one checkpoint path")
        return cls(
            [
                PolyT5PropertyPredictor.from_checkpoint(
                    path, tokenizer_path, device=device, **kwargs
                )
                for path in paths
            ]
        )

    def __len__(self) -> int:
        """Return the number of members."""
        return len(self.predictors)

    def predict_with_uncertainty(
        self, pselfies: Sequence[str]
    ) -> list[tuple[float, float, int]]:
        """Score candidates and report the spread across members.

        Args:
            pselfies: Candidates as PSELFIES or PSMILES -- each member
                auto-detects the notation exactly as
                :meth:`PolyT5PropertyPredictor.__call__` does.

        Returns:
            One ``(mean, std, n_members)`` triple per input, in input order.
            ``std`` is the population standard deviation over the members that
            actually returned a number, so it is ``0.0`` for a single
            contributing member. A candidate no member could score is
            ``(nan, nan, 0)``.
        """
        per_member = [list(predictor(pselfies)) for predictor in self.predictors]

        summaries: list[tuple[float, float, int]] = []
        for position in range(len(pselfies)):
            finite = [
                float(member[position])
                for member in per_member
                if position < len(member) and _is_finite(member[position])
            ]
            if not finite:
                summaries.append((NON_NUMERIC_VALUE, NON_NUMERIC_VALUE, 0))
                continue
            mean = sum(finite) / len(finite)
            variance = sum((value - mean) ** 2 for value in finite) / len(finite)
            summaries.append((mean, math.sqrt(variance), len(finite)))
        return summaries

    def __call__(self, candidates: Sequence[str]) -> list[float]:
        """Score candidates, returning the ensemble mean. **The injection point.**

        Satisfies the same ``Callable[[Sequence[str]], Sequence[float]]``
        contract as :meth:`PolyT5PropertyPredictor.__call__`, so an ensemble
        drops straight into
        :func:`polyt5.evaluation.evaluate_generation`'s ``property_predictor``.

        Args:
            candidates: Polymers as PSELFIES or PSMILES.

        Returns:
            One ``float`` per input: the mean over the members that produced a
            number, or :data:`NON_NUMERIC_VALUE` when none did.
        """
        return [mean for mean, _, _ in self.predict_with_uncertainty(candidates)]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_members={len(self.predictors)})"


class CachedPredictor:
    """Memoises any predictor callable, so re-scoring a candidate is free.

    The RL loop re-scores the same polymers many times -- across GRPO group
    members, across epochs, and whenever a candidate reappears in a batch -- and
    a beam-search forward pass is by far the most expensive part of a reward.
    This wrapper keeps an LRU cache in front of it.

    Cache keys are the *input strings*, whitespace-stripped, not chemically
    canonicalised. Two different writings of one polymer are different model
    inputs and may legitimately receive different predictions, so collapsing
    them by default would change results rather than merely speed them up. A
    caller who wants chemical identity can pass
    ``key_fn=polyt5.chemistry.canonical_psmiles``.

    Example:
        >>> cached = CachedPredictor(predictor)          # doctest: +SKIP
        >>> report = evaluate_generation(                # doctest: +SKIP
        ...     candidates, target_property=500.0, property_predictor=cached
        ... )
        >>> cached.cache_info()                          # doctest: +SKIP
        CacheInfo(hits=812, misses=188, maxsize=100000, currsize=188)
    """

    def __init__(
        self,
        predictor: Callable[[Sequence[str]], Sequence[float]],
        *,
        maxsize: int | None = 100_000,
        key_fn: Callable[[str], str | None] | None = None,
    ) -> None:
        """Wrap a predictor.

        Args:
            predictor: Any ``Callable[[Sequence[str]], Sequence[float]]``,
                including :class:`PolyT5PropertyPredictor`.
            maxsize: Maximum number of cached candidates; ``None`` is unbounded.
                The least recently used entry is evicted when the cache is full.
            key_fn: Maps a candidate to its cache key. Defaults to
                ``str.strip``. A key of ``None`` disables caching for that
                candidate, which is what a failed canonicalisation should do.

        Raises:
            ValueError: If ``maxsize`` is not ``None`` and is less than 1.
        """
        if maxsize is not None and maxsize < 1:
            raise ValueError(f"maxsize must be >= 1 or None, got {maxsize}")
        self._predictor = predictor
        self._maxsize = maxsize
        self._key_fn = key_fn or (lambda text: text.strip())
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def __call__(self, candidates: Sequence[str]) -> list[float]:
        """Score candidates, consulting the cache first.

        Args:
            candidates: Polymers to score, in whatever notation the wrapped
                predictor accepts.

        Returns:
            One ``float`` per input, in input order -- identical to what the
            wrapped predictor would have returned, cached entries included.
        """
        keys: list[str | None] = []
        for entry in candidates:
            try:
                keys.append(self._key_fn(entry) if isinstance(entry, str) else None)
            except Exception:
                keys.append(None)

        values: list[float | None] = [None] * len(candidates)
        pending: dict[str, list[int]] = {}
        for position, key in enumerate(keys):
            if key is not None and key in self._cache:
                self._cache.move_to_end(key)
                values[position] = self._cache[key]
                self._hits += 1
                continue
            self._misses += 1
            if key is None:
                pending.setdefault(f"\x00uncacheable:{position}", []).append(position)
            else:
                pending.setdefault(key, []).append(position)

        if pending:
            order = list(pending)
            computed = list(self._predictor([candidates[pending[key][0]] for key in order]))
            for key, value in zip(order, computed, strict=True):
                as_float = float(value)
                for position in pending[key]:
                    values[position] = as_float
                if not key.startswith("\x00uncacheable:"):
                    self._store(key, as_float)

        return [NON_NUMERIC_VALUE if value is None else value for value in values]

    def _store(self, key: str, value: float) -> None:
        """Insert a value, evicting the least recently used entry when full."""
        self._cache[key] = value
        self._cache.move_to_end(key)
        if self._maxsize is not None:
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def cache_info(self) -> CacheInfo:
        """Return hit/miss counters and the current cache occupancy.

        Counters are per LOOKUP, as in :func:`functools.lru_cache`, so
        ``hits + misses`` is the number of candidates ever passed in. A batch
        that repeats a not-yet-cached candidate records two misses but still
        invokes the wrapped predictor once, because a batch is deduplicated
        before it is forwarded; ``misses`` is therefore an upper bound on the
        number of forward passes, not a count of them.

        Returns:
            A :class:`CacheInfo` with the same field names as
            :func:`functools.lru_cache`'s.
        """
        return CacheInfo(
            hits=self._hits,
            misses=self._misses,
            maxsize=self._maxsize,
            currsize=len(self._cache),
        )

    def clear(self) -> None:
        """Drop every cached value and reset the counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def __repr__(self) -> str:
        info = self.cache_info()
        return (
            f"{type(self).__name__}({self._predictor!r}, hits={info.hits}, "
            f"misses={info.misses}, currsize={info.currsize})"
        )


def _is_finite(value: Any) -> bool:
    """Return whether ``value`` is a real, finite number (never ``bool``)."""
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
