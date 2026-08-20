"""T5 building-block layers: RMS layer norm, feed-forward, encoder/decoder layers.

All sublayers use T5's pre-norm residual structure:
``x = x + dropout(sublayer(layer_norm(x)))``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from polyt5.model.attention import MultiHeadAttention
from polyt5.model.config import PolyT5Config


class T5LayerNorm(nn.Module):
    """T5's RMS-style layer norm: NO mean subtraction and NO bias.

    Only the root-mean-square of the activations is normalized; a learned
    per-channel scale is applied afterwards. Variance is accumulated in
    float32 for stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """Create the scale parameter.

        Args:
            hidden_size: Size of the normalized (last) dimension.
            eps: Numerical-stability epsilon inside the square root.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Normalize by RMS over the last dimension and rescale."""
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.type_as(self.weight)


class FeedForward(nn.Module):
    """T5 feed-forward block: dense-relu-dense (v1.0) or gated-GELU (v1.1).

    ``"relu"``: ``wo(dropout(relu(wi(x))))``.
    ``"gated-gelu"``: ``wo(dropout(gelu(wi_0(x)) * wi_1(x)))``.
    All projections are bias-free, per T5.
    """

    def __init__(self, config: PolyT5Config) -> None:
        """Build projections for the configured variant.

        Args:
            config: Model configuration (reads ``feed_forward_proj``).
        """
        super().__init__()
        self.feed_forward_proj = config.feed_forward_proj
        if config.feed_forward_proj == "gated-gelu":
            self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
            self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        else:  # "relu"
            self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply the feed-forward transform."""
        if self.feed_forward_proj == "gated-gelu":
            hidden = nn.functional.gelu(self.wi_0(hidden_states)) * self.wi_1(hidden_states)
        else:
            hidden = nn.functional.relu(self.wi(hidden_states))
        return self.wo(self.dropout(hidden))


class EncoderLayer(nn.Module):
    """One encoder layer: pre-normed bidirectional self-attention, then FFN."""

    def __init__(self, config: PolyT5Config, *, has_relative_attention_bias: bool) -> None:
        """Build sublayers.

        Args:
            config: Model configuration.
            has_relative_attention_bias: True only for the stack's first
                layer, which owns the shared relative position bias table.
        """
        super().__init__()
        self.self_attn_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.self_attn = MultiHeadAttention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,
            bidirectional=True,
        )
        self.ffn_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ffn = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_bias: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Run the layer.

        Args:
            hidden_states: ``(batch, seq, d_model)`` input.
            attention_mask: Additive mask broadcastable to attention scores.
            position_bias: Shared relative position bias from layer 0
                (computed here when ``None`` and this layer owns the table).

        Returns:
            Tuple of ``(hidden_states, position_bias)`` — the bias is
            returned so the stack can pass it to subsequent layers.
        """
        normed = self.self_attn_layer_norm(hidden_states)
        attn_out, position_bias, _ = self.self_attn(
            normed, mask=attention_mask, position_bias=position_bias
        )
        hidden_states = hidden_states + self.dropout(attn_out)

        normed = self.ffn_layer_norm(hidden_states)
        hidden_states = hidden_states + self.dropout(self.ffn(normed))
        return hidden_states, position_bias


class DecoderLayer(nn.Module):
    """One decoder layer: causal self-attention -> cross-attention -> FFN.

    Each sublayer is pre-normed with its own :class:`T5LayerNorm`.
    Cross-attention carries NO relative position bias (implicit zeros).
    """

    def __init__(self, config: PolyT5Config, *, has_relative_attention_bias: bool) -> None:
        """Build sublayers.

        Args:
            config: Model configuration.
            has_relative_attention_bias: True only for the stack's first
                layer, which owns the shared relative position bias table.
        """
        super().__init__()
        self.self_attn_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.self_attn = MultiHeadAttention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,
            bidirectional=False,
        )
        self.cross_attn_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.cross_attn = MultiHeadAttention(
            config, has_relative_attention_bias=False, bidirectional=False
        )
        self.ffn_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ffn = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        self_attention_mask: Tensor | None = None,
        cross_attention_mask: Tensor | None = None,
        position_bias: Tensor | None = None,
        past_key_value: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[
        Tensor,
        Tensor | None,
        tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]] | None,
    ]:
        """Run the layer.

        Args:
            hidden_states: ``(batch, tgt_len, d_model)`` decoder input.
            encoder_hidden_states: ``(batch, src_len, d_model)`` encoder
                output attended to by cross-attention.
            self_attention_mask: Additive causal (+ padding) mask for
                self-attention.
            cross_attention_mask: Additive encoder padding mask for
                cross-attention.
            position_bias: Shared causal relative position bias from layer 0.
            past_key_value: ``(self_attn_kv, cross_attn_kv)`` cache from the
                previous decoding step.
            use_cache: Whether to return an updated cache.

        Returns:
            Tuple of ``(hidden_states, position_bias, present_key_value)``.
        """
        self_past = past_key_value[0] if past_key_value is not None else None
        cross_past = past_key_value[1] if past_key_value is not None else None

        normed = self.self_attn_layer_norm(hidden_states)
        attn_out, position_bias, self_present = self.self_attn(
            normed,
            mask=self_attention_mask,
            position_bias=position_bias,
            past_key_value=self_past,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + self.dropout(attn_out)

        normed = self.cross_attn_layer_norm(hidden_states)
        cross_out, _, cross_present = self.cross_attn(
            normed,
            key_value_states=encoder_hidden_states,
            mask=cross_attention_mask,
            past_key_value=cross_past,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + self.dropout(cross_out)

        normed = self.ffn_layer_norm(hidden_states)
        hidden_states = hidden_states + self.dropout(self.ffn(normed))

        present = (self_present, cross_present) if use_cache else None
        return hidden_states, position_bias, present
