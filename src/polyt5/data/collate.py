"""Batch collators for polyT5 pretraining and fine-tuning.

``pad_sequences`` is pure python (testable without torch); the collator
classes import torch lazily inside ``__call__`` so this module can be imported
while torch is still installing.

Label padding uses ``-100`` -- the paper explicitly states padded label
positions are set to -100 so the cross-entropy loss ignores them. Encoder
input padding uses ``pad_id`` with ``attention_mask`` 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from polyt5.data.span_corruption import (
    DEFAULT_CONFIG,
    SpanCorruptionConfig,
    span_corrupt,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps torch lazy at runtime
    import torch

__all__ = ["LABEL_IGNORE_ID", "Seq2SeqCollator", "SpanCorruptionCollator", "pad_sequences"]

#: Label id ignored by torch's cross-entropy loss (paper: "-100").
LABEL_IGNORE_ID = -100


def _require_torch() -> Any:
    """Import torch on first use so module import never needs it."""
    import torch

    return torch


def pad_sequences(
    seqs: Sequence[Sequence[int]],
    pad_id: int,
    max_length: int | None = None,
    pad_to_multiple_of: int | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """Truncate and right-pad sequences to a common length.

    Args:
        seqs: Token id sequences of varying length.
        pad_id: Fill value for padded positions.
        max_length: If given, each sequence is first truncated (from the tail)
            to this length; the padded width therefore never exceeds
            ``max_length`` unless ``pad_to_multiple_of`` rounds it up.
        pad_to_multiple_of: If given, the common width is rounded up to the
            next multiple (useful for tensor-core efficiency).

    Returns:
        ``(padded, attention_mask)`` as plain nested lists -- ``mask[i][j]``
        is 1 for real tokens and 0 for padding.
    """
    trimmed = [list(s)[:max_length] if max_length is not None else list(s) for s in seqs]
    width = max((len(s) for s in trimmed), default=0)
    if pad_to_multiple_of and width % pad_to_multiple_of:
        width += pad_to_multiple_of - width % pad_to_multiple_of
    padded = [s + [pad_id] * (width - len(s)) for s in trimmed]
    mask = [[1] * len(s) + [0] * (width - len(s)) for s in trimmed]
    return padded, mask


class SpanCorruptionCollator:
    """Torch collator applying span corruption to a batch of raw sequences.

    Each raw sequence is truncated to ``max_length`` FIRST and then corrupted,
    so sentinels can never be cut away from their spans; the corrupted encoder
    input is always <= ``max_length`` because spans only shrink the sequence.

    Reseeding rule (documented for resumable training): the rng for batch
    ``b`` of epoch ``e`` is ``np.random.default_rng([seed, e, b])``.
    ``set_epoch(e)`` sets the epoch and resets the batch counter to 0, so
    resuming at epoch ``e`` and replaying batches in the same order reproduces
    the exact same masks. With ``seed=None`` a fresh OS-entropy generator is
    drawn per batch, so every call (and hence every epoch) masks differently.
    """

    def __init__(
        self,
        sentinel_ids: Sequence[int],
        pad_id: int,
        eos_id: int | None,
        config: SpanCorruptionConfig = DEFAULT_CONFIG,
        max_length: int = 200,
        pad_to_multiple_of: int | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the collator.

        Args:
            sentinel_ids: Sentinel token ids, index ``n`` == ``<extra_id_n>``.
            pad_id: Encoder padding token id.
            eos_id: EOS token id (protected from masking, appended to labels);
                ``None`` disables both behaviors.
            config: Span-corruption hyperparameters.
            max_length: Encoder truncation length (paper: 200).
            pad_to_multiple_of: Optional width rounding for padded tensors.
            seed: Base seed for reproducible masking; ``None`` -> fresh
                entropy per batch.
        """
        self.sentinel_ids = [int(s) for s in sentinel_ids]
        self.pad_id = int(pad_id)
        self.eos_id = None if eos_id is None else int(eos_id)
        self.config = config
        self.max_length = int(max_length)
        self.pad_to_multiple_of = pad_to_multiple_of
        self.seed = seed
        self._epoch = 0
        self._batch_index = 0

    def set_epoch(self, epoch: int) -> None:
        """Advance to ``epoch``, resetting the batch counter (see class doc)."""
        self._epoch = int(epoch)
        self._batch_index = 0

    def _next_rng(self) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        rng = np.random.default_rng([self.seed, self._epoch, self._batch_index])
        self._batch_index += 1
        return rng

    def __call__(self, batch: Sequence[Sequence[int]]) -> dict[str, torch.Tensor]:
        """Corrupt, pad and tensorize a batch of raw token id sequences.

        Returns:
            ``{"input_ids", "attention_mask", "labels"}`` as long tensors.
            Rows whose sequence was too short to corrupt have all-``-100``
            labels (no loss signal); with the paper's length-200 inputs this
            never occurs, but callers filtering degenerate rows can check it.
        """
        torch = _require_torch()
        rng = self._next_rng()
        results = [
            span_corrupt(
                list(seq)[: self.max_length],
                sentinel_ids=self.sentinel_ids,
                config=self.config,
                rng=rng,
                eos_id=self.eos_id,
            )
            for seq in batch
        ]
        inputs, mask = pad_sequences(
            [r.input_ids for r in results],
            pad_id=self.pad_id,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        labels, _ = pad_sequences(
            [r.labels for r in results],
            pad_id=LABEL_IGNORE_ID,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class Seq2SeqCollator:
    """Torch collator for supervised fine-tuning -- no masking at all."""

    def __init__(
        self,
        pad_id: int,
        max_source_length: int = 200,
        max_target_length: int = 200,
    ) -> None:
        """Initialize the collator.

        Args:
            pad_id: Encoder padding token id (labels always pad with -100).
            max_source_length: Source truncation length.
            max_target_length: Target truncation length.
        """
        self.pad_id = int(pad_id)
        self.max_source_length = int(max_source_length)
        self.max_target_length = int(max_target_length)

    def __call__(
        self, batch: Sequence[tuple[list[int], list[int]]]
    ) -> dict[str, torch.Tensor]:
        """Pad and tensorize ``(source_ids, target_ids)`` pairs.

        Returns:
            ``{"input_ids", "attention_mask", "labels"}`` as long tensors;
            labels are padded with -100 so padded positions carry no loss.
        """
        torch = _require_torch()
        sources = [src for src, _ in batch]
        targets = [tgt for _, tgt in batch]
        inputs, mask = pad_sequences(
            sources, pad_id=self.pad_id, max_length=self.max_source_length
        )
        labels, _ = pad_sequences(
            targets, pad_id=LABEL_IGNORE_ID, max_length=self.max_target_length
        )
        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
