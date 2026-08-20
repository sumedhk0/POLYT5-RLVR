"""T5 relative position bucketing and bias.

Faithful port of the bucketing scheme from the original T5 (Raffel et al.,
2020) as implemented in ``transformers.models.t5.modeling_t5`` — we
reimplement it in plain PyTorch here; there is no transformers dependency.

Semantics (must match HF exactly):
    * Relative position is ``key_position - query_position``.
    * ``bidirectional=True`` (encoder): the bucket space is halved; negative
      relative positions (key before query) use the lower half and positive
      relative positions get a ``num_buckets // 2`` offset.
    * ``bidirectional=False`` (decoder self-attention): positive relative
      positions (future keys) all collapse to bucket 0; only distances into
      the past are distinguished.
    * Within each direction, the first half of the available buckets covers
      exact small distances one-to-one; the second half is logarithmically
      spaced up to ``max_distance``. Distances at or beyond ``max_distance``
      clamp into the final bucket.

Ordering contract: the resulting bias is ADDED TO THE RAW ATTENTION SCORES
BEFORE SOFTMAX, and the attention mask is combined ADDITIVELY with those
same scores (masked positions get a large negative value) — i.e.
``softmax(q @ k^T + position_bias + mask)``. See
:class:`polyt5.model.attention.MultiHeadAttention`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def relative_position_bucket(
    relative_position: Tensor,
    *,
    bidirectional: bool,
    num_buckets: int,
    max_distance: int,
) -> Tensor:
    """Map integer relative positions to bucket indices, T5-style.

    Args:
        relative_position: Integer tensor of ``key_pos - query_pos`` values
            (any shape).
        bidirectional: True for encoder self-attention (distinguish sign of
            the offset), False for causal decoder self-attention.
        num_buckets: Total number of buckets.
        max_distance: Distances >= this saturate into the final bucket.

    Returns:
        Long tensor of bucket indices in ``[0, num_buckets)`` with the same
        shape as ``relative_position``.
    """
    relative_buckets = torch.zeros_like(relative_position, dtype=torch.long)
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        # Causal: future keys (positive relative position) clamp to distance 0.
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    # relative_position is now a non-negative distance.

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # Larger distances are bucketed logarithmically up to max_distance.
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )

    relative_buckets += torch.where(is_small, relative_position.long(), relative_position_if_large)
    return relative_buckets


class RelativePositionBias(nn.Module):
    """Learned relative position bias, one scalar per (bucket, head).

    Only the FIRST layer of each stack owns an instance of this module (HF's
    ``has_relative_attention_bias`` pattern); the computed bias tensor is
    passed down to all subsequent layers of the stack. Cross-attention uses
    NO relative position bias at all (HF T5 uses an implicit zero bias
    there).

    The returned bias is added to the pre-softmax attention scores together
    (additively) with the attention mask; see the module docstring.
    """

    def __init__(
        self,
        num_buckets: int,
        max_distance: int,
        num_heads: int,
        bidirectional: bool,
    ) -> None:
        """Initialize the bias table.

        Args:
            num_buckets: Number of relative-position buckets.
            max_distance: Saturation distance for the logarithmic buckets.
            num_heads: Number of attention heads (one bias scalar per head).
            bidirectional: Whether the owning attention is bidirectional.
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def forward(self, query_length: int, key_length: int, device: torch.device) -> Tensor:
        """Compute the position bias for a (query_length, key_length) grid.

        Args:
            query_length: Number of query positions.
            key_length: Number of key positions.
            device: Device on which to build the position grid.

        Returns:
            Float tensor of shape ``(1, num_heads, query_length, key_length)``
            that broadcasts against ``(batch, num_heads, q, k)`` attention
            scores.
        """
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # (q, k)
        buckets = relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(buckets)  # (q, k, num_heads)
        return values.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, q, k)
