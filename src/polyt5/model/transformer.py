"""PolyT5 encoder/decoder stacks and the conditional-generation model.

A faithful from-scratch (plain PyTorch, no transformers dependency)
reimplementation of ``T5ForConditionalGeneration`` semantics, sized per
Table S2 of Sahu et al. (npj Artificial Intelligence 2026).

The paper does not state the weight initialization; we use T5's original
scheme (see :meth:`PolyT5ForConditionalGeneration._init_weights`), scaled by
``config.initializer_factor``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from polyt5.model.attention import MultiHeadAttention
from polyt5.model.config import PolyT5Config
from polyt5.model.layers import DecoderLayer, EncoderLayer, FeedForward, T5LayerNorm
from polyt5.model.relative_position import RelativePositionBias

# Type alias: per-layer decoder cache — (self_attn (k, v), cross_attn (k, v)).
LayerCache = tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]


def _masked_fill_value(dtype: torch.dtype) -> float:
    """Large negative additive-mask value for ``dtype`` (HF convention)."""
    return torch.finfo(dtype).min


def _expand_padding_mask(attention_mask: Tensor, dtype: torch.dtype) -> Tensor:
    """Convert a ``(batch, k_len)`` 1/0 mask to an additive ``(batch, 1, 1, k_len)`` mask.

    Args:
        attention_mask: 1 where tokens are real, 0 where padded.
        dtype: Dtype of the attention scores the mask will be added to.

    Returns:
        Float mask with 0.0 at visible positions and a large negative value
        at masked positions.
    """
    expanded = attention_mask[:, None, None, :].to(dtype)
    return (1.0 - expanded) * _masked_fill_value(dtype)


def _causal_mask(
    query_length: int, past_length: int, dtype: torch.dtype, device: torch.device
) -> Tensor:
    """Build an additive causal mask of shape ``(1, 1, q_len, past + q_len)``.

    Query position ``i`` (offset by ``past_length`` cached steps) may attend
    key positions ``j <= past_length + i``.
    """
    key_length = past_length + query_length
    query_pos = torch.arange(query_length, device=device)[:, None] + past_length
    key_pos = torch.arange(key_length, device=device)[None, :]
    allowed = key_pos <= query_pos  # (q, k)
    mask = torch.zeros(query_length, key_length, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, _masked_fill_value(dtype))
    return mask[None, None, :, :]


class PolyT5Stack(nn.Module):
    """Encoder or decoder stack sharing the model-level token embedding.

    Layer 0's self-attention owns the stack's relative position bias table;
    the computed bias tensor is threaded through the remaining layers. A
    final :class:`T5LayerNorm` and dropout close the stack (T5 structure).
    """

    def __init__(
        self, config: PolyT5Config, embed_tokens: nn.Embedding, *, is_decoder: bool
    ) -> None:
        """Build the stack.

        Args:
            config: Model configuration.
            embed_tokens: Token embedding shared with the sibling stack (and
                possibly the LM head).
            is_decoder: Whether this stack is the (causal, cross-attending)
                decoder.
        """
        super().__init__()
        self.config = config
        self.is_decoder = is_decoder
        self.embed_tokens = embed_tokens
        num_layers = config.num_decoder_layers_resolved if is_decoder else config.num_layers
        layer_cls = DecoderLayer if is_decoder else EncoderLayer
        self.layers = nn.ModuleList(
            [layer_cls(config, has_relative_attention_bias=(i == 0)) for i in range(num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        past_key_values: tuple[LayerCache, ...] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[LayerCache, ...] | None]:
        """Run the stack.

        Args:
            input_ids: ``(batch, seq)`` token ids.
            attention_mask: ``(batch, seq)`` 1/0 padding mask over
                ``input_ids`` (for a cached decoder, over the full sequence
                including cached steps).
            encoder_hidden_states: Encoder output for the decoder's
                cross-attention (decoder only).
            encoder_attention_mask: ``(batch, src_len)`` 1/0 padding mask
                over the encoder sequence (decoder only).
            past_key_values: Per-layer decoder caches from previous steps.
            use_cache: Whether to return updated per-layer caches.

        Returns:
            Tuple of ``(hidden_states, present_key_values)`` with
            ``hidden_states`` of shape ``(batch, seq, d_model)``.
        """
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        dtype = hidden_states.dtype
        device = hidden_states.device
        q_len = input_ids.shape[1]

        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0][0].shape[2]

        if self.is_decoder:
            # Causal mask (offset by cached length), combined additively with
            # any decoder padding mask.
            self_mask = _causal_mask(q_len, past_length, dtype, device)
            if attention_mask is not None:
                self_mask = self_mask + _expand_padding_mask(attention_mask, dtype)
            cross_mask = (
                _expand_padding_mask(encoder_attention_mask, dtype)
                if encoder_attention_mask is not None
                else None
            )
        else:
            self_mask = (
                _expand_padding_mask(attention_mask, dtype) if attention_mask is not None else None
            )
            cross_mask = None

        position_bias: Tensor | None = None
        presents: list[LayerCache] = []
        for i, layer in enumerate(self.layers):
            if self.is_decoder:
                layer_past = past_key_values[i] if past_key_values is not None else None
                hidden_states, position_bias, present = layer(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    self_attention_mask=self_mask,
                    cross_attention_mask=cross_mask,
                    position_bias=position_bias,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                )
                if use_cache:
                    presents.append(present)
            else:
                hidden_states, position_bias = layer(
                    hidden_states,
                    attention_mask=self_mask,
                    position_bias=position_bias,
                )

        hidden_states = self.dropout(self.final_layer_norm(hidden_states))
        return hidden_states, tuple(presents) if use_cache else None


@dataclass
class Seq2SeqLMOutput:
    """Output container for :class:`PolyT5ForConditionalGeneration.forward`.

    Attributes:
        loss: Cross-entropy loss (``ignore_index=-100``) when labels are
            given, else ``None``.
        logits: ``(batch, tgt_len, vocab_size)`` LM logits.
        encoder_last_hidden_state: Final encoder hidden states, if the
            encoder ran (or was provided).
        past_key_values: Per-layer decoder caches when ``use_cache=True``.
    """

    loss: Tensor | None
    logits: Tensor
    encoder_last_hidden_state: Tensor | None
    past_key_values: tuple[LayerCache, ...] | None


class PolyT5ForConditionalGeneration(nn.Module):
    """From-scratch T5 encoder-decoder with an LM head (HF-equivalent semantics).

    The token embedding is shared between encoder and decoder, and — when
    ``config.tie_word_embeddings`` — with the LM head as well. When tied,
    the decoder output is rescaled by ``d_model ** -0.5`` before the LM head
    projection (T5 trains with this rescaling because the tied embedding is
    initialized at unit scale; HF does exactly the same).
    """

    def __init__(self, config: PolyT5Config) -> None:
        """Build the model and apply T5's weight initialization.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = PolyT5Stack(config, self.shared, is_decoder=False)
        self.decoder = PolyT5Stack(config, self.shared, is_decoder=True)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # [AMBIGUITY] The paper does not state the initialization; we use T5's
        # original scheme (see _init_weights), scaled by initializer_factor.
        self.apply(self._init_weights)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.shared.weight

    def _init_weights(self, module: nn.Module) -> None:
        """T5's initializer scheme, scaled by ``config.initializer_factor``.

        * Shared embedding and (untied) LM head: ``N(0, factor * 1.0)``.
        * Relative position bias tables: ``N(0, factor * d_model**-0.5)``.
        * Attention: q ``N(0, factor * (d_model * d_kv)**-0.5)``; k and v
          ``N(0, factor * d_model**-0.5)``; o
          ``N(0, factor * (num_heads * d_kv)**-0.5)``. The missing
          ``1/sqrt(d_kv)`` score scaling is folded into q's init.
        * Feed-forward: input projections ``N(0, factor * d_model**-0.5)``;
          output projection ``N(0, factor * d_ff**-0.5)``.
        * :class:`T5LayerNorm` weights: ``factor * 1.0``.
        """
        factor = self.config.initializer_factor
        d_model = self.config.d_model
        d_kv = self.config.d_kv
        d_ff = self.config.d_ff
        num_heads = self.config.num_heads

        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, PolyT5ForConditionalGeneration):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, RelativePositionBias):
            module.relative_attention_bias.weight.data.normal_(
                mean=0.0, std=factor * (d_model**-0.5)
            )
        elif isinstance(module, MultiHeadAttention):
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * d_kv) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((num_heads * d_kv) ** -0.5))
        elif isinstance(module, FeedForward):
            for name in ("wi", "wi_0", "wi_1"):
                if hasattr(module, name):
                    getattr(module, name).weight.data.normal_(
                        mean=0.0, std=factor * (d_model**-0.5)
                    )
            module.wo.weight.data.normal_(mean=0.0, std=factor * (d_ff**-0.5))

    def _shift_right(self, labels: Tensor) -> Tensor:
        """Build decoder inputs from labels (the HF ``_shift_right`` contract).

        Labels are shifted one position right, ``decoder_start_token_id`` is
        prepended, and any ``-100`` sentinel in the shifted inputs is
        replaced with ``pad_token_id``.

        Args:
            labels: ``(batch, tgt_len)`` target ids (may contain -100).

        Returns:
            ``(batch, tgt_len)`` decoder input ids.
        """
        shifted = labels.new_zeros(labels.shape)
        shifted[..., 1:] = labels[..., :-1].clone()
        shifted[..., 0] = self.config.decoder_start_token_id
        shifted.masked_fill_(shifted == -100, self.config.pad_token_id)
        return shifted

    def encode(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Run the encoder only (used by the future RL sampling loop).

        Args:
            input_ids: ``(batch, src_len)`` source token ids.
            attention_mask: ``(batch, src_len)`` 1/0 padding mask.

        Returns:
            ``(batch, src_len, d_model)`` encoder hidden states.
        """
        hidden_states, _ = self.encoder(input_ids, attention_mask=attention_mask)
        return hidden_states

    def decode_step(
        self,
        decoder_input_ids: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor | None = None,
        past_key_values: tuple[LayerCache, ...] | None = None,
    ) -> tuple[Tensor, tuple[LayerCache, ...]]:
        """Cache-friendly single decoding step (for future RL sampling; no RL here).

        Args:
            decoder_input_ids: ``(batch, new_len)`` newest decoder token(s)
                only — cached steps must not be re-passed.
            encoder_hidden_states: Output of :meth:`encode`.
            encoder_attention_mask: Source padding mask.
            past_key_values: Cache returned by the previous call.

        Returns:
            Tuple of ``(logits, present_key_values)`` with logits of shape
            ``(batch, new_len, vocab_size)``.
        """
        hidden_states, presents = self.decoder(
            decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = self._project_to_vocab(hidden_states)
        assert presents is not None
        return logits, presents

    def _project_to_vocab(self, hidden_states: Tensor) -> Tensor:
        """Apply the (possibly tied) LM head, with T5's tied-weight rescaling."""
        if self.config.tie_word_embeddings:
            # T5 rescales the LM-head input by d_model**-0.5 when the head is
            # tied to the unit-scale token embedding (HF does the same).
            hidden_states = hidden_states * (self.config.d_model**-0.5)
        return self.lm_head(hidden_states)

    def forward(
        self,
        input_ids: Tensor | None,
        attention_mask: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        encoder_outputs: Tensor | None = None,
        past_key_values: tuple[LayerCache, ...] | None = None,
        use_cache: bool = False,
    ) -> Seq2SeqLMOutput:
        """Full seq2seq forward pass.

        Args:
            input_ids: ``(batch, src_len)`` source ids; may be ``None`` when
                ``encoder_outputs`` is provided.
            attention_mask: ``(batch, src_len)`` 1/0 source padding mask.
            decoder_input_ids: ``(batch, tgt_len)`` decoder input ids. When
                ``None`` and ``labels`` is given, they are built by
                right-shifting the labels (see :meth:`_shift_right`).
            decoder_attention_mask: ``(batch, tgt_len)`` 1/0 decoder padding
                mask (combined with the causal mask).
            labels: ``(batch, tgt_len)`` target ids with ``-100`` at ignored
                positions; enables the loss.
            encoder_outputs: Precomputed encoder hidden states (skips the
                encoder).
            past_key_values: Decoder caches for incremental decoding.
            use_cache: Whether to return updated decoder caches.

        Returns:
            A :class:`Seq2SeqLMOutput`.

        Raises:
            ValueError: If neither encoder inputs nor decoder inputs can be
                determined.
        """
        if encoder_outputs is None:
            if input_ids is None:
                raise ValueError("Either input_ids or encoder_outputs is required.")
            encoder_outputs = self.encode(input_ids, attention_mask=attention_mask)

        if decoder_input_ids is None:
            if labels is None:
                raise ValueError("Either decoder_input_ids or labels is required.")
            decoder_input_ids = self._shift_right(labels)

        decoder_hidden, presents = self.decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self._project_to_vocab(decoder_hidden)

        loss: Tensor | None = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            encoder_last_hidden_state=encoder_outputs,
            past_key_values=presents,
        )

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters, counting tied weights once.

        Args:
            trainable_only: Count only parameters with ``requires_grad``.

        Returns:
            Total parameter count.
        """
        params = (p for p in self.parameters() if p.requires_grad or not trainable_only)
        # dict keyed by data_ptr de-duplicates tied parameters.
        unique = {p.data_ptr(): p for p in params}
        return sum(p.numel() for p in unique.values())
