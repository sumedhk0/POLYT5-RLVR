"""Torch ``Dataset`` wrappers over prepared PSELFIES corpora.

Both datasets take an INJECTED duck-typed tokenizer (anything exposing
``encode(text, add_eos=..., max_length=..., truncation=...) -> list[int]``)
and never construct one, so they stay decoupled from the concurrently
developed ``polyt5.tokenization`` package. They hold only plain picklable
state (lists, ints, the tokenizer object itself), which makes them safe for
``DataLoader(num_workers > 0)`` as long as the tokenizer pickles.

Items are raw ``list[int]`` (or pairs of them); padding and span-corruption
masking are the collators' job (``polyt5.data.collate``).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch.utils.data

__all__ = ["PSelfiesCorpus", "Seq2SeqDataset"]


class PSelfiesCorpus(torch.utils.data.Dataset):
    """Tokenized PSELFIES corpus for span-corruption pretraining.

    Each item is the token-id list of one PSELFIES string; the
    ``SpanCorruptionCollator`` performs masking and padding downstream.

    Args:
        pselfies: Prepared PSELFIES strings (one polymer each).
        tokenizer: Duck-typed tokenizer with an ``encode`` method.
        max_length: Truncation length in tokens (paper: 200).
        add_eos: Append the tokenizer's EOS token to each encoding.
    """

    def __init__(
        self,
        pselfies: Sequence[str],
        tokenizer,
        *,
        max_length: int = 200,
        add_eos: bool = True,
    ) -> None:
        self._pselfies = list(pselfies)
        self._tokenizer = tokenizer
        self._max_length = int(max_length)
        self._add_eos = bool(add_eos)

    def __len__(self) -> int:
        return len(self._pselfies)

    def __getitem__(self, index: int) -> list[int]:
        """Encode one PSELFIES string to a token-id list."""
        return self._tokenizer.encode(
            self._pselfies[index],
            add_eos=self._add_eos,
            max_length=self._max_length,
            truncation=True,
        )

    @property
    def stats(self) -> dict:
        """Cheap descriptive statistics for run manifests."""
        return {
            "n_examples": len(self._pselfies),
            "max_length": self._max_length,
            "add_eos": self._add_eos,
        }


class Seq2SeqDataset(torch.utils.data.Dataset):
    """Supervised (source, target) pairs for fine-tuning.

    Each item is ``(source_ids, target_ids)``; the ``Seq2SeqCollator`` pads
    and builds labels downstream. EOS is always appended to both sides.

    Args:
        pairs: ``(source_text, target_text)`` tuples, e.g. the output of
            ``build_prediction_examples`` / ``build_generation_examples``.
        tokenizer: Duck-typed tokenizer with an ``encode`` method.
        max_source_length: Truncation length for sources (paper: 200).
        max_target_length: Truncation length for targets (paper: 200).
    """

    def __init__(
        self,
        pairs: Sequence[tuple[str, str]],
        tokenizer,
        *,
        max_source_length: int = 200,
        max_target_length: int = 200,
    ) -> None:
        self._pairs = [(str(s), str(t)) for s, t in pairs]
        self._tokenizer = tokenizer
        self._max_source_length = int(max_source_length)
        self._max_target_length = int(max_target_length)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        """Encode one example to ``(source_ids, target_ids)``."""
        source, target = self._pairs[index]
        source_ids = self._tokenizer.encode(
            source, add_eos=True, max_length=self._max_source_length, truncation=True
        )
        target_ids = self._tokenizer.encode(
            target, add_eos=True, max_length=self._max_target_length, truncation=True
        )
        return source_ids, target_ids

    @property
    def stats(self) -> dict:
        """Cheap descriptive statistics for run manifests."""
        return {
            "n_examples": len(self._pairs),
            "max_source_length": self._max_source_length,
            "max_target_length": self._max_target_length,
        }
