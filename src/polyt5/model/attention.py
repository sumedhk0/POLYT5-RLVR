"""T5-style multi-head attention.

Two deliberate — and famous — deviations from the vanilla Transformer,
both inherited from the original T5:

    1. NO bias terms on the q/k/v/o projections.
    2. NO ``1/sqrt(d_kv)`` scaling of the dot product. T5 folds the scale
       into the weight initialization instead (the query projection is
       initialized with std ``(d_model * d_kv) ** -0.5``), so the raw
       ``q @ k^T`` scores are used directly.

Score assembly, in order: ``scores = q @ k^T``; then the relative position
bias (when present) and the additive attention mask (0 for visible, a large
negative number for masked) are ADDED to the scores; softmax follows.
Cross-attention gets NO relative position bias (HF T5 uses an implicit zero
bias there).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from polyt5.model.config import PolyT5Config
from polyt5.model.relative_position import RelativePositionBias


class MultiHeadAttention(nn.Module):
    """Multi-head attention used for self-, causal self-, and cross-attention.

    Only the FIRST layer of each stack is constructed with
    ``has_relative_attention_bias=True`` and owns a
    :class:`RelativePositionBias`; it computes the bias tensor once and the
    stack passes it down to every subsequent layer (HF's
    ``has_relative_attention_bias`` sharing pattern).

    Supports incremental decoding through ``past_key_value``:
        * self-attention: cached keys/values are concatenated with the new
          step's keys/values;
        * cross-attention: keys/values are computed once from the encoder
          output and reused verbatim on every step.
    """

    def __init__(
        self,
        config: PolyT5Config,
        *,
        has_relative_attention_bias: bool,
        bidirectional: bool,
    ) -> None:
        """Build the projection layers (all bias-free, per T5).

        Args:
            config: Model configuration.
            has_relative_attention_bias: Whether this instance owns the
                relative position bias table (first layer of a stack only).
            bidirectional: Whether the owned bias (if any) is bidirectional.
        """
        super().__init__()
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.num_heads = config.num_heads
        self.inner_dim = config.num_heads * config.d_kv

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.has_relative_attention_bias = has_relative_attention_bias
        if has_relative_attention_bias:
            self.relative_attention_bias: RelativePositionBias | None = RelativePositionBias(
                num_buckets=config.relative_attention_num_buckets,
                max_distance=config.relative_attention_max_distance,
                num_heads=config.num_heads,
                bidirectional=bidirectional,
            )
        else:
            self.relative_attention_bias = None

    def _shape(self, x: Tensor, batch: int) -> Tensor:
        """Reshape ``(batch, seq, inner)`` to ``(batch, heads, seq, d_kv)``."""
        return x.view(batch, -1, self.num_heads, self.d_kv).transpose(1, 2)

    def _unshape(self, x: Tensor, batch: int) -> Tensor:
        """Reshape ``(batch, heads, seq, d_kv)`` back to ``(batch, seq, inner)``."""
        return x.transpose(1, 2).contiguous().view(batch, -1, self.inner_dim)

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Tensor | None = None,
        mask: Tensor | None = None,
        position_bias: Tensor | None = None,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor | None, tuple[Tensor, Tensor] | None]:
        """Run attention.

        Args:
            hidden_states: Query source, ``(batch, q_len, d_model)``.
            key_value_states: Encoder output for cross-attention; ``None``
                for self-attention.
            mask: Additive attention mask broadcastable to
                ``(batch, num_heads, q_len, k_len)`` — 0 where visible, a
                large negative number where masked. Combined additively with
                the position bias before softmax.
            position_bias: Precomputed relative position bias to reuse
                (passed down from the stack's first layer). Computed here
                when ``None`` and this layer owns the bias table; otherwise
                zeros (cross-attention has no position bias).
            past_key_value: Cached ``(key, value)`` from previous decoding
                steps.
            use_cache: Whether to return an updated ``(key, value)`` cache.

        Returns:
            Tuple of ``(output, position_bias, present_key_value)`` where
            ``output`` is ``(batch, q_len, d_model)``, ``position_bias`` is
            the (possibly newly computed) bias for downstream layers, and
            ``present_key_value`` is the updated cache (``None`` when
            ``use_cache`` is False).
        """
        batch, q_len, _ = hidden_states.shape
        is_cross = key_value_states is not None

        query = self._shape(self.q(hidden_states), batch)  # (b, h, q, d_kv)

        if is_cross:
            if past_key_value is not None:
                key, value = past_key_value  # encoder keys/values never change
            else:
                key = self._shape(self.k(key_value_states), batch)
                value = self._shape(self.v(key_value_states), batch)
        else:
            key = self._shape(self.k(hidden_states), batch)
            value = self._shape(self.v(hidden_states), batch)
            if past_key_value is not None:
                key = torch.cat([past_key_value[0], key], dim=2)
                value = torch.cat([past_key_value[1], value], dim=2)

        key_length = key.shape[2]

        # NOTE: no 1/sqrt(d_kv) scaling — see module docstring.
        scores = torch.matmul(query, key.transpose(3, 2))  # (b, h, q, k)

        if position_bias is None:
            if self.relative_attention_bias is not None:
                # With a cache, queries are the LAST q_len positions of the
                # full sequence: compute the bias over the full key length and
                # slice the trailing query rows (HF behavior).
                full_bias = self.relative_attention_bias(
                    key_length, key_length, device=scores.device
                )
                position_bias = full_bias[:, :, -q_len:, :]
            else:
                # Cross-attention / no bias table: implicit zero bias.
                position_bias = torch.zeros(
                    (1, self.num_heads, q_len, key_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )

        scores = scores + position_bias
        if mask is not None:
            scores = scores + mask  # additive: masked positions get a large negative value

        weights = torch.softmax(scores.float(), dim=-1).type_as(scores)
        weights = self.dropout(weights)

        output = self.o(self._unshape(torch.matmul(weights, value), batch))
        present: tuple[Tensor, Tensor] | None = (key, value) if use_cache else None
        return output, position_bias, present
