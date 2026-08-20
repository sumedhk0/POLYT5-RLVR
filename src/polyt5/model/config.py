"""Configuration for the from-scratch PolyT5 (T5) encoder-decoder model.

Paper-specified architecture values come from Table S2 of Sahu et al.,
"polyT5" (npj Artificial Intelligence 2026): ``d_model``, ``num_layers``
(shared by encoder and decoder), ``d_ff``, ``num_heads``, ``d_kv`` and
``n_positions`` (200 for every size; the paper states "All models employ
relative positional encodings with a maximum input length of 200 tokens").

Everything else is NOT stated in the paper and is defaulted to the
HuggingFace ``T5Config`` defaults, marked with ``# [AMBIGUITY]`` below.

Note on relative-position saturation: with ``n_positions=200`` and the HF
default ``relative_attention_max_distance=128``, relative distances beyond
128 all saturate into the final logarithmic bucket, so tokens more than 128
positions apart share one position bias. The paper does not address this
interaction; we keep the HF default rather than invent a value.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

_ALLOWED_FF_PROJ = ("relu", "gated-gelu")


@dataclass
class PolyT5Config:
    """Hyperparameters for :class:`polyt5.model.PolyT5ForConditionalGeneration`.

    Attributes:
        vocab_size: Token vocabulary size. The tokenizer track supplies real
            ids at runtime; this is just an integer here.
        d_model: Hidden size of embeddings and residual stream (Table S2).
        d_kv: Per-head key/value dimension (Table S2). T5 does NOT require
            ``d_kv * num_heads == d_model``.
        num_heads: Number of attention heads (Table S2).
        d_ff: Feed-forward inner dimension (Table S2).
        num_layers: Number of encoder layers (Table S2; the paper's
            ``num_layers`` applies to both encoder and decoder).
        num_decoder_layers: Decoder depth; ``None`` means "same as
            ``num_layers``" (the paper implies symmetry).
        n_positions: Maximum input length in tokens (200 in the paper).
        relative_attention_num_buckets: Bucket count for relative position
            bias.
        relative_attention_max_distance: Log-bucketing saturation distance.
        dropout_rate: Dropout probability.
        layer_norm_epsilon: Epsilon inside T5's RMS layer norm.
        initializer_factor: Global multiplier on T5's init std devs.
        feed_forward_proj: ``"relu"`` (T5 v1.0) or ``"gated-gelu"`` (v1.1).
        tie_word_embeddings: Share the LM head with the token embedding.
        pad_token_id: Padding id (HF T5 convention: 0).
        eos_token_id: End-of-sequence id (HF T5 convention: 1).
        decoder_start_token_id: First decoder input token when shifting
            labels right (HF T5 uses the pad id, 0).
    """

    vocab_size: int = 458
    d_model: int = 256
    d_kv: int = 64
    num_heads: int = 4
    d_ff: int = 1024
    num_layers: int = 4
    num_decoder_layers: int | None = None
    n_positions: int = 200
    # [AMBIGUITY] Not in the paper; HF T5Config default (32 buckets).
    relative_attention_num_buckets: int = 32
    # [AMBIGUITY] Not in the paper; HF default 128 (< n_positions=200 -> saturation, see module
    # docstring).
    relative_attention_max_distance: int = 128
    # [AMBIGUITY] Not in the paper; HF T5Config default 0.1.
    dropout_rate: float = 0.1
    # [AMBIGUITY] Not in the paper; HF T5Config default 1e-6.
    layer_norm_epsilon: float = 1e-6
    # [AMBIGUITY] Not in the paper; HF T5Config default 1.0.
    initializer_factor: float = 1.0
    # [AMBIGUITY] The paper never says T5 v1.0 (relu) vs v1.1 (gated-gelu). "relu" reproduces the
    # Table S2 parameter counts exactly, so it is the default.
    feed_forward_proj: str = "relu"
    # [AMBIGUITY] Not in the paper; HF T5 default is tied. Tying reproduces Table S2 counts.
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    # [AMBIGUITY] Not in the paper; HF T5 starts decoding from the pad token (id 0).
    decoder_start_token_id: int = 0

    def __post_init__(self) -> None:
        """Validate field combinations.

        Raises:
            ValueError: If ``feed_forward_proj`` is not one of the two
                supported values.
        """
        if self.feed_forward_proj not in _ALLOWED_FF_PROJ:
            raise ValueError(
                f"feed_forward_proj must be one of {_ALLOWED_FF_PROJ}, "
                f"got {self.feed_forward_proj!r}"
            )
        if self.d_kv * self.num_heads != self.d_model:
            # T5 explicitly allows this (inner projection dim != d_model), but it is unusual
            # enough to flag.
            warnings.warn(
                f"d_kv * num_heads = {self.d_kv * self.num_heads} != d_model = {self.d_model}; "
                "T5 allows this, but double-check it is intentional.",
                stacklevel=2,
            )

    @property
    def num_decoder_layers_resolved(self) -> int:
        """Decoder depth, defaulting to ``num_layers`` when unset."""
        return self.num_layers if self.num_decoder_layers is None else self.num_decoder_layers

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation of this config."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolyT5Config:
        """Build a config from a dict, rejecting unknown keys.

        Args:
            data: Mapping of field names to values.

        Returns:
            A validated :class:`PolyT5Config`.

        Raises:
            ValueError: If ``data`` contains keys that are not config fields.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PolyT5Config:
        """Load a config from a YAML file.

        Args:
            path: Path to a YAML mapping of config fields.

        Returns:
            The parsed :class:`PolyT5Config`.
        """
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a YAML mapping in {path}, got {type(data).__name__}")
        return cls.from_dict(data)

    def save_yaml(self, path: str | Path) -> None:
        """Write this config to ``path`` as YAML.

        Args:
            path: Destination file path.
        """
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def estimate_num_parameters(self) -> int:
        """Analytically count model parameters for this config.

        Mirrors :meth:`PolyT5ForConditionalGeneration.num_parameters` exactly
        (used in tests to cross-check the paper's Table S2 totals).

        Returns:
            Total trainable parameter count.
        """
        inner = self.num_heads * self.d_kv
        attn = 4 * self.d_model * inner  # q, k, v, o projections, all bias-free
        if self.feed_forward_proj == "gated-gelu":
            ffn = 3 * self.d_model * self.d_ff  # wi_0, wi_1, wo
        else:
            ffn = 2 * self.d_model * self.d_ff  # wi, wo
        ln = self.d_model  # RMS norm: weight only, no bias

        enc_layer = attn + ffn + 2 * ln
        dec_layer = 2 * attn + ffn + 3 * ln
        rel_bias = self.relative_attention_num_buckets * self.num_heads  # first layer of each stack

        encoder = self.num_layers * enc_layer + rel_bias + ln  # + final layer norm
        decoder = self.num_decoder_layers_resolved * dec_layer + rel_bias + ln

        total = self.vocab_size * self.d_model + encoder + decoder  # shared embedding
        if not self.tie_word_embeddings:
            total += self.d_model * self.vocab_size  # separate LM head
        return total
