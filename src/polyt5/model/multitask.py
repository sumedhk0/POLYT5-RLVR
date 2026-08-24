"""One pretrained encoder, a regression head, descriptor heads, and the decoder.

Today prediction and generation are separate fine-tunes from the same
pretrained checkpoint -- siblings that share nothing after pretraining, so
anything learned about what makes a polymer high-Tg is invisible to the
generator. That separation follows the paper training them as distinct tasks,
not any property of T5, which is a multi-task architecture by design::

    pretrained polyT5 encoder      (shared; gradients from both tasks)
    |-- regression head   -> Tg as a scalar           (prediction)
    |-- descriptor heads  -> the 100 LamaLab features (auxiliary)
    `-- decoder           -> PSELFIES                 (generation)

The wrapper OWNS the backbone rather than subclassing it, so:

* ``state_dict`` keys are namespaced under ``backbone.``. A Group A checkpoint
  therefore fails loudly in :class:`polyt5.inference.PolyT5PropertyPredictor`
  instead of half-loading -- Group A produces new models ALONGSIDE the existing
  ones and must never be mistaken for them.
* :attr:`config` still returns the backbone's :class:`PolyT5Config`, so
  ``Trainer.save`` records a ``model_config`` that rebuilds the backbone.
* :meth:`forward_generation` delegates unchanged. Generation is untouched.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn

from polyt5.model.config import PolyT5Config
from polyt5.model.heads import (
    RegressionHead,
    masked_mean_pool,
    weighted_huber_loss,
    weighted_lm_loss,
)
from polyt5.model.transformer import PolyT5ForConditionalGeneration, Seq2SeqLMOutput

__all__ = ["HeadOutput", "MultiTaskConfig", "PolyT5MultiTask"]


@dataclass(frozen=True)
class MultiTaskConfig:
    """Which heads exist and how their losses combine.

    Attributes:
        use_regression_head: Attach the scalar Tg head. When ``False`` the Tg
            objective is the backbone's text decode, as in the baseline.
        n_descriptors: Width of the auxiliary descriptor head; ``0`` disables it.
            This is the number of columns the train-split standardizer KEPT,
            not the raw 100.
        descriptor_lambda: Weight of the descriptor term in
            ``L = L_Tg + lambda * L_descriptors``. 100 auxiliary targets against
            one Tg target can swamp the objective we care about, so this is
            configurable and its sensitivity is reported.
        huber_delta: Huber transition point, in standardised units.
        head_dropout: Dropout on the pooled vector before each head.
    """

    use_regression_head: bool = False
    n_descriptors: int = 0
    descriptor_lambda: float = 0.1
    huber_delta: float = 1.0
    head_dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, stored in the run config."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MultiTaskConfig:
        """Rebuild from :meth:`to_dict` output, ignoring unknown keys."""
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in known})


@dataclass
class HeadOutput:
    """What one prediction-side forward pass produced.

    Attributes:
        loss: The combined objective, or ``None`` when no targets were given.
        tg_standardised: ``(batch,)`` head output in standardised units, or
            ``None`` on the text path (where Tg is decoded, not regressed).
        tg_kelvin: ``tg_standardised`` mapped back to Kelvin, or ``None``.
        descriptors: ``(batch, n_descriptors)`` head output, or ``None``.
        tg_loss: The Tg term alone.
        descriptor_loss: The descriptor term alone, BEFORE ``descriptor_lambda``.
    """

    loss: Tensor | None
    tg_standardised: Tensor | None
    tg_kelvin: Tensor | None
    descriptors: Tensor | None
    tg_loss: Tensor | None
    descriptor_loss: Tensor | None


class PolyT5MultiTask(nn.Module):
    """A polyT5 backbone plus the Group A prediction heads."""

    def __init__(
        self, backbone: PolyT5ForConditionalGeneration, config: MultiTaskConfig
    ) -> None:
        """Wrap ``backbone``.

        Args:
            backbone: A pretrained or freshly built conditional-generation model.
            config: Which heads to attach and how to combine their losses.
        """
        super().__init__()
        self.backbone = backbone
        self.head_config = config
        d_model = backbone.config.d_model
        self.tg_head = (
            RegressionHead(d_model, 1, dropout=config.head_dropout)
            if config.use_regression_head
            else None
        )
        self.descriptor_head = (
            RegressionHead(d_model, config.n_descriptors, dropout=config.head_dropout)
            if config.n_descriptors > 0
            else None
        )
        # Buffers, so the target scaling travels inside state_dict and an
        # inference-time inverse transform can never silently use the wrong
        # numbers.
        self.register_buffer("tg_mean", torch.zeros(()))
        self.register_buffer("tg_std", torch.ones(()))

    @property
    def config(self) -> PolyT5Config:
        """The BACKBONE's config, so a saved checkpoint can rebuild the model."""
        return self.backbone.config

    def set_target_scaling(self, mean: float, std: float) -> None:
        """Record the train-split target statistics used to invert predictions.

        Args:
            mean: Train-split mean Tg in Kelvin.
            std: Train-split standard deviation in Kelvin.

        Raises:
            ValueError: If ``std`` is not finite and positive.
        """
        if not math.isfinite(std) or std <= 0.0:
            raise ValueError(f"target std must be finite and > 0, got {std}")
        with torch.no_grad():
            self.tg_mean.fill_(float(mean))
            self.tg_std.fill_(float(std))

    def _pooled(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return masked_mean_pool(
            self.backbone.encode(input_ids, attention_mask=attention_mask), attention_mask
        )

    def _descriptor_term(
        self,
        pooled: Tensor,
        descriptor_targets: Tensor | None,
        weights: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Return ``(head_output, unweighted_loss)`` for the descriptor head."""
        if self.descriptor_head is None:
            return None, None
        predictions = self.descriptor_head(pooled)
        if descriptor_targets is None:
            return predictions, None
        loss = weighted_huber_loss(
            predictions,
            descriptor_targets.to(predictions.dtype),
            delta=self.head_config.huber_delta,
            weights=weights,
        )
        return predictions, loss

    def forward_regression(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        tg_targets: Tensor | None = None,
        descriptor_targets: Tensor | None = None,
        weights: Tensor | None = None,
    ) -> HeadOutput:
        """Predict Tg with the scalar head.

        Args:
            input_ids: ``(batch, src_len)`` PSELFIES token ids.
            attention_mask: ``(batch, src_len)`` 1/0 padding mask.
            tg_targets: ``(batch,)`` STANDARDISED targets; enables the loss.
            descriptor_targets: ``(batch, n_descriptors)`` standardised targets.
            weights: ``(batch,)`` per-example weights.

        Returns:
            A :class:`HeadOutput`.

        Raises:
            RuntimeError: If no regression head was built.
        """
        if self.tg_head is None:
            raise RuntimeError(
                "forward_regression needs MultiTaskConfig(use_regression_head=True); this "
                "model has no scalar head, so its Tg objective is the text decode"
            )
        pooled = self._pooled(input_ids, attention_mask)
        standardised = self.tg_head(pooled).squeeze(-1)
        descriptors, descriptor_loss = self._descriptor_term(
            pooled, descriptor_targets, weights
        )

        tg_loss: Tensor | None = None
        total: Tensor | None = None
        if tg_targets is not None:
            tg_loss = weighted_huber_loss(
                standardised,
                tg_targets.to(standardised.dtype),
                delta=self.head_config.huber_delta,
                weights=weights,
            )
            total = tg_loss
        if descriptor_loss is not None:
            scaled = self.head_config.descriptor_lambda * descriptor_loss
            total = scaled if total is None else total + scaled

        return HeadOutput(
            loss=total,
            tg_standardised=standardised,
            tg_kelvin=standardised * self.tg_std + self.tg_mean,
            descriptors=descriptors,
            tg_loss=tg_loss,
            descriptor_loss=descriptor_loss,
        )

    def forward_text(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        *,
        descriptor_targets: Tensor | None = None,
        weights: Tensor | None = None,
    ) -> HeadOutput:
        """Predict Tg as TEXT, the baseline objective, plus optional extras.

        With neither ``descriptor_targets`` nor ``weights``, the returned loss
        is the backbone's own token-mean cross-entropy, unchanged -- that is how
        arm B0 reproduces the frozen baseline rather than approximating it.

        Args:
            input_ids: ``(batch, src_len)`` PSELFIES token ids.
            attention_mask: ``(batch, src_len)`` 1/0 padding mask.
            labels: ``(batch, tgt_len)`` numeric-string target ids, ``-100`` padded.
            descriptor_targets: ``(batch, n_descriptors)`` standardised targets.
            weights: ``(batch,)`` per-example weights.

        Returns:
            A :class:`HeadOutput` whose ``tg_standardised`` and ``tg_kelvin``
            are ``None`` -- on this path Tg is decoded, not regressed.
        """
        output = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        tg_loss = (
            output.loss
            if weights is None
            else weighted_lm_loss(output.logits, labels, weights=weights)
        )

        descriptors: Tensor | None = None
        descriptor_loss: Tensor | None = None
        if self.descriptor_head is not None:
            pooled = masked_mean_pool(output.encoder_last_hidden_state, attention_mask)
            descriptors, descriptor_loss = self._descriptor_term(
                pooled, descriptor_targets, weights
            )

        total = tg_loss
        if descriptor_loss is not None:
            total = total + self.head_config.descriptor_lambda * descriptor_loss

        return HeadOutput(
            loss=total,
            tg_standardised=None,
            tg_kelvin=None,
            descriptors=descriptors,
            tg_loss=tg_loss,
            descriptor_loss=descriptor_loss,
        )

    def forward_generation(
        self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor
    ) -> Seq2SeqLMOutput:
        """Tg-conditioned generation, delegated to the backbone untouched.

        Args:
            input_ids: ``(batch, src_len)`` conditioning-number token ids.
            attention_mask: ``(batch, src_len)`` 1/0 padding mask.
            labels: ``(batch, tgt_len)`` PSELFIES target ids, ``-100`` padded.

        Returns:
            The backbone's :class:`~polyt5.model.transformer.Seq2SeqLMOutput`.
        """
        return self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    @torch.no_grad()
    def predict_tg(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Predict Tg in KELVIN, with the train-split scaling inverted.

        Args:
            input_ids: ``(batch, src_len)`` PSELFIES token ids.
            attention_mask: ``(batch, src_len)`` 1/0 padding mask.

        Returns:
            ``(batch,)`` predictions in Kelvin.
        """
        was_training = self.training
        self.eval()
        try:
            output = self.forward_regression(input_ids, attention_mask)
        finally:
            self.train(was_training)
        assert output.tg_kelvin is not None  # forward_regression always sets it
        return output.tg_kelvin

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters, counting tied weights once.

        Args:
            trainable_only: Count only parameters with ``requires_grad``.

        Returns:
            Total parameter count across backbone and heads.
        """
        params = (p for p in self.parameters() if p.requires_grad or not trainable_only)
        unique = {p.data_ptr(): p for p in params}
        return sum(p.numel() for p in unique.values())
