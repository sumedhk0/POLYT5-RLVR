"""Score PSELFIES with the Group A regression head instead of decoding text.

The baseline predictor generates the number one character at a time under beam
search, so a decode can fail to be a number at all. The regression head cannot:
it emits one float per input. ``non_numeric_rate`` is therefore structurally
0.0 for this predictor, and it is still reported -- an honest zero is better
than an omitted column that invites the reader to carry the old number over.

A Group A checkpoint deliberately does NOT load in
:class:`polyt5.inference.PolyT5PropertyPredictor`: its ``state_dict`` keys are
namespaced under ``backbone.`` and it carries head metadata the baseline loader
knows nothing about. Group A produces new models ALONGSIDE the existing five,
never replacements, and a silent half-load into a reward path is the one
failure this design refuses to allow.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from polyt5.chemistry import psmiles_to_pselfies
from polyt5.data.prepare import format_property_value
from polyt5.inference.predictor import (
    DEFAULT_MAX_SOURCE_LENGTH,
    NON_NUMERIC_VALUE,
    PredictionResult,
    looks_like_pselfies,
)
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.model.multitask import MultiTaskConfig, PolyT5MultiTask
from polyt5.tokenization import PolyT5Tokenizer
from polyt5.training import load_checkpoint
from polyt5.utils import select_device

__all__ = ["GROUP_A_CONFIG_KEY", "RegressionPropertyPredictor"]

#: Run-config key under which a Group A checkpoint stores its head metadata.
GROUP_A_CONFIG_KEY = "group_a"


class RegressionPropertyPredictor:
    """Load a Group A checkpoint and score PSELFIES with its regression head."""

    def __init__(
        self,
        model: PolyT5MultiTask,
        tokenizer: PolyT5Tokenizer,
        *,
        device: str = "auto",
        batch_size: int = 64,
        max_source_length: int = DEFAULT_MAX_SOURCE_LENGTH,
        property_name: str | None = None,
    ) -> None:
        """Wrap a loaded model for inference.

        Args:
            model: A :class:`~polyt5.model.multitask.PolyT5MultiTask` whose
                target scaling has been set.
            tokenizer: The tokenizer the model was trained with.
            device: ``"auto"``, ``"cpu"``, ``"cuda"``, or an explicit device.
            batch_size: Candidates per forward pass. A throughput knob only:
                the head is independent per row, so batching must not change
                any prediction.
            max_source_length: Input truncation length. # [PAPER] 200. Clamped
                to the model's ``n_positions`` when that is smaller.
            property_name: Optional label ("Tg") carried for bookkeeping.

        Raises:
            ValueError: If ``batch_size`` is below 1, or the model has no
                regression head.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if model.tg_head is None:
            raise ValueError(
                "this model has no regression head; use PolyT5PropertyPredictor for a "
                "text-decoding checkpoint"
            )
        self.device = select_device("auto") if device == "auto" else str(device)
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.property_name = property_name

        n_positions = getattr(model.config, "n_positions", None)
        self.max_source_length = (
            min(int(max_source_length), int(n_positions))
            if isinstance(n_positions, int) and n_positions > 0
            else int(max_source_length)
        )
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path | None = None,
        *,
        device: str = "auto",
        allow_unverified_tokenizer: bool = False,
        **kwargs: Any,
    ) -> RegressionPropertyPredictor:
        """Rebuild a predictor from a Group A checkpoint.

        Args:
            checkpoint_path: A ``.pt`` file written during a Group A run.
            tokenizer_path: Tokenizer artifact; defaults to the path recorded
                inside the checkpoint.
            device: Passed to :meth:`__init__`.
            allow_unverified_tokenizer: Permit a checkpoint with no recorded
                ``tokenizer_sha256``. Off by default: a wrong vocabulary yields
                plausible wrong NUMBERS rather than a crash.
            **kwargs: Forwarded to :meth:`__init__`.

        Returns:
            A ready-to-use predictor in ``eval`` mode.

        Raises:
            ValueError: If no tokenizer can be located or verified, or the
                checkpoint carries no ``group_a`` head metadata.
        """
        checkpoint_path = Path(checkpoint_path)
        payload = load_checkpoint(checkpoint_path, map_location="cpu")

        resolved = tokenizer_path or payload.get("tokenizer_path")
        if resolved is None:
            raise ValueError(
                f"{checkpoint_path} records no tokenizer_path and none was supplied"
            )
        tokenizer = PolyT5Tokenizer.from_file(Path(resolved))
        recorded_sha = payload.get("tokenizer_sha256")
        if recorded_sha is None:
            if not allow_unverified_tokenizer:
                raise ValueError(
                    f"{checkpoint_path} recorded no tokenizer_sha256, so the vocabulary "
                    "cannot be verified; pass allow_unverified_tokenizer=True to accept it"
                )
        elif recorded_sha != tokenizer.sha256:
            raise ValueError(
                "tokenizer mismatch: the checkpoint was trained with vocabulary "
                f"{recorded_sha[:16]} but the supplied tokenizer is "
                f"{tokenizer.sha256[:16]}"
            )

        group_a = (payload.get("config") or {}).get(GROUP_A_CONFIG_KEY)
        if not isinstance(group_a, dict) or "heads" not in group_a:
            raise ValueError(
                f"{checkpoint_path} carries no {GROUP_A_CONFIG_KEY!r}.heads metadata, so its "
                "head widths are unknown; it is not a Group A checkpoint"
            )

        backbone = PolyT5ForConditionalGeneration(PolyT5Config.from_dict(payload["model_config"]))
        model = PolyT5MultiTask(backbone, MultiTaskConfig.from_dict(group_a["heads"]))
        model.load_state_dict(payload["model_state"])
        return cls(model, tokenizer, device=device, **kwargs)

    @torch.no_grad()
    def predict(self, pselfies: Sequence[str]) -> list[PredictionResult]:
        """Score PSELFIES strings with the regression head.

        Args:
            pselfies: Candidate polymers as PSELFIES. Entries that are not
                non-empty strings are reported as non-numeric without reaching
                the model.

        Returns:
            One :class:`~polyt5.inference.PredictionResult` per input, in input
            order. Never raises on model output.
        """
        results: list[PredictionResult | None] = [None] * len(pselfies)
        live: list[tuple[int, str]] = []
        for position, entry in enumerate(pselfies):
            if isinstance(entry, str) and entry.strip():
                live.append((position, entry))
            else:
                results[position] = PredictionResult(
                    source=entry if isinstance(entry, str) else str(entry),
                    decoded="", value=None, is_numeric=False,
                )

        for start in range(0, len(live), self.batch_size):
            chunk = live[start : start + self.batch_size]
            encoded = self.tokenizer.batch_encode(
                [text for _, text in chunk], add_eos=True,
                max_length=self.max_source_length, padding=True, truncation=True,
            )
            values = self.model.predict_tg(
                torch.tensor(encoded["input_ids"], device=self.device),
                torch.tensor(encoded["attention_mask"], device=self.device),
            )
            for (position, text), value in zip(chunk, values.tolist(), strict=True):
                finite = bool(value == value and abs(value) != float("inf"))
                results[position] = PredictionResult(
                    source=text,
                    decoded=format_property_value(value) if finite else "",
                    value=float(value) if finite else None,
                    is_numeric=finite,
                )
        return [result for result in results if result is not None]

    def predict_values(self, pselfies: Sequence[str]) -> list[float | None]:
        """Score PSELFIES and return values only, ``None`` for failures."""
        return [result.value for result in self.predict(pselfies)]

    def __call__(self, candidates: Sequence[str]) -> list[float]:
        """Injection point for :func:`polyt5.evaluation.evaluate_generation`.

        Mirrors :meth:`polyt5.inference.PolyT5PropertyPredictor.__call__`
        exactly, because that function hands its predictor the *canonical
        PSMILES* of the screened candidates while the model is trained on
        PSELFIES: notation is auto-detected per entry (an all-bracket-token
        string is used as PSELFIES, anything else is converted from PSMILES),
        and the whole scoring call is wrapped so an inference failure degrades
        to NaNs for this batch rather than raising into the evaluation run.
        Skipping either step would silently score PSMILES input as a
        mis-tokenized, wrong-but-finite Kelvin value instead of converting or
        flagging it -- exactly the failure mode this predictor exists to avoid.

        Args:
            candidates: Polymers as PSELFIES or PSMILES, in any mixture.

        Returns:
            One float per candidate; a failure is
            :data:`~polyt5.inference.NON_NUMERIC_VALUE` (NaN), which the TP
            metric drops from both numerator and denominator.
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

    def __repr__(self) -> str:
        return (
            f"RegressionPropertyPredictor(property_name={self.property_name!r}, "
            f"device={self.device!r}, batch_size={self.batch_size})"
        )
