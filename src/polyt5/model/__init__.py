"""From-scratch T5 encoder-decoder for the polyT5 reproduction (plain PyTorch).

Public API re-exports; see the individual modules for details.
"""

from __future__ import annotations

from polyt5.model.attention import MultiHeadAttention
from polyt5.model.config import PolyT5Config
from polyt5.model.heads import (
    RegressionHead,
    masked_mean_pool,
    weighted_huber_loss,
    weighted_lm_loss,
)
from polyt5.model.layers import DecoderLayer, EncoderLayer, FeedForward, T5LayerNorm
from polyt5.model.relative_position import RelativePositionBias, relative_position_bucket
from polyt5.model.transformer import (
    PolyT5ForConditionalGeneration,
    PolyT5Stack,
    Seq2SeqLMOutput,
)

__all__ = [
    "MultiHeadAttention",
    "PolyT5Config",
    "RegressionHead",
    "masked_mean_pool",
    "weighted_huber_loss",
    "weighted_lm_loss",
    "DecoderLayer",
    "EncoderLayer",
    "FeedForward",
    "T5LayerNorm",
    "RelativePositionBias",
    "relative_position_bucket",
    "PolyT5ForConditionalGeneration",
    "PolyT5Stack",
    "Seq2SeqLMOutput",
]
