"""Encoder pooling, the regression head, and the two weighted losses.

Tg is currently emitted as TEXT, one character at a time under beam search: the
model must learn digit place-value, can emit non-numeric output, and token
cross-entropy gives it no sense that 236 and 237 are adjacent while 236 and 936
are not. A scalar head on the pooled encoder state removes all three problems.

Design points, each from the Group A spec:

* **Masked mean pooling, not first-token.** T5 has no ``[CLS]`` and never
  trained one, so position 0 carries no summary. The mean is taken over
  non-pad positions only, and an all-pad row pools to zero rather than dividing
  by zero.
* **Huber, not MSE.** Some Tg labels carry up to 145 K of experimental spread.
  Squaring that lets a handful of near-noise labels dominate the gradient.
* **Per-example weights apply after the per-example reduction.** For the
  100-column descriptor head that means the column mean first, then the weight
  -- otherwise a reliability weight would be applied 100 times.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = [
    "RegressionHead",
    "masked_mean_pool",
    "weighted_huber_loss",
    "weighted_lm_loss",
]


def masked_mean_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean-pool encoder states over non-pad positions.

    Args:
        hidden_states: ``(batch, seq, d_model)`` encoder output.
        attention_mask: ``(batch, seq)`` 1/0 mask; 1 marks a real token.

    Returns:
        ``(batch, d_model)``. A row with no unmasked position pools to zeros
        rather than NaN -- a degenerate input is not a crash.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class RegressionHead(nn.Module):
    """Dropout + linear projection from the pooled encoder state.

    Used twice: width 1 for Tg, width ``n_kept_descriptors`` for the auxiliary
    descriptor targets.
    """

    def __init__(self, d_model: int, n_outputs: int = 1, *, dropout: float = 0.1) -> None:
        """Build the head.

        Args:
            d_model: Encoder hidden size.
            n_outputs: Number of scalar outputs.
            dropout: Dropout applied to the pooled vector.

        Raises:
            ValueError: If ``d_model`` or ``n_outputs`` is below 1.
        """
        super().__init__()
        if d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {d_model}")
        if n_outputs < 1:
            raise ValueError(f"n_outputs must be >= 1, got {n_outputs}")
        self.n_outputs = int(n_outputs)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, self.n_outputs)
        # Match T5's scheme for a projection out of the residual stream, and
        # start from a zero bias so an untrained head predicts the standardised
        # mean (0.0), i.e. the train mean in Kelvin.
        nn.init.normal_(self.projection.weight, mean=0.0, std=d_model**-0.5)
        nn.init.zeros_(self.projection.bias)

    def forward(self, pooled: Tensor) -> Tensor:
        """Project the pooled state.

        Args:
            pooled: ``(batch, d_model)`` output of :func:`masked_mean_pool`.

        Returns:
            ``(batch, n_outputs)``.
        """
        return self.projection(self.dropout(pooled))


def _apply_example_weights(per_example: Tensor, weights: Tensor | None) -> Tensor:
    """Reduce per-example losses to a scalar, optionally weighted."""
    if weights is None:
        return per_example.mean()
    weight = weights.to(per_example.dtype)
    denominator = weight.sum().clamp(min=torch.finfo(per_example.dtype).eps)
    return (per_example * weight).sum() / denominator


def weighted_huber_loss(
    predictions: Tensor,
    targets: Tensor,
    *,
    delta: float = 1.0,
    weights: Tensor | None = None,
) -> Tensor:
    """Huber loss, meaned over any trailing dimensions, then weighted per example.

    Args:
        predictions: ``(batch,)`` or ``(batch, n_outputs)``.
        targets: Same shape as ``predictions``.
        delta: Huber transition point, in standardised units.
        weights: Optional ``(batch,)`` per-example weights.

    Returns:
        A scalar tensor.
    """
    per_element = nn.functional.huber_loss(
        predictions, targets, reduction="none", delta=delta
    )
    per_example = (
        per_element
        if per_element.dim() <= 1
        else per_element.mean(dim=tuple(range(1, per_element.dim())))
    )
    return _apply_example_weights(per_example, weights)


def weighted_lm_loss(
    logits: Tensor, labels: Tensor, *, weights: Tensor | None = None
) -> Tensor:
    """Token cross-entropy, meaned within each example, then weighted per example.

    With ``weights=None`` and equal token counts per row this equals the
    backbone's own token-mean cross-entropy; with unequal counts it is the
    example-mean instead. Callers that must reproduce the baseline exactly use
    the backbone's loss, not this function -- see
    :meth:`polyt5.model.multitask.PolyT5MultiTask.forward_text`.

    Args:
        logits: ``(batch, tgt_len, vocab)``.
        labels: ``(batch, tgt_len)`` with ``-100`` at ignored positions.
        weights: Optional ``(batch,)`` per-example weights.

    Returns:
        A scalar tensor.
    """
    per_token = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(labels.shape)
    valid = (labels != -100).to(per_token.dtype)
    per_example = (per_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
    return _apply_example_weights(per_example, weights)
