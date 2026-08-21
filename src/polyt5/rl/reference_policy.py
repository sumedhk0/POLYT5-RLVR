"""The frozen reference policy that anchors GRPO's KL term.

A copy of the supervised checkpoint, loaded once and never trained again, used
only to SCORE sequences the current policy sampled -- never to generate. If it
ever received gradient, the KL anchor (see :func:`polyt5.rl.grpo.k3_kl`) would
be measuring the policy's drift from a moving target instead of a fixed one,
and the whole run would be silently wasted.

"Genuinely frozen" here means three things at once, not one:

1. ``.eval()`` -- dropout and any other train-time-only behaviour is off.
2. ``requires_grad_(False)`` on every parameter -- so even a caller who
   forgets to wrap a call in ``torch.no_grad()`` cannot build a graph through
   this model.
3. :meth:`ReferencePolicy.score` itself runs under ``torch.no_grad()`` --
   defense in depth on top of (2), not a substitute for it.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.training import load_checkpoint

__all__ = ["ReferencePolicy"]


class ReferencePolicy:
    """Wraps a model as a frozen, score-only reference policy."""

    def __init__(
        self,
        model: PolyT5ForConditionalGeneration,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        """Freeze ``model`` in place and hold it for scoring.

        Mutates ``model`` directly (moves it, calls ``.eval()``, clears
        ``requires_grad`` on every parameter) rather than copying it, so a
        caller who already holds a reference to the same module object sees
        it become frozen too.

        Args:
            model: The model to freeze and wrap.
            device: Device to run scoring on.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> ReferencePolicy:
        """Build a frozen reference policy from a saved checkpoint.

        Args:
            path: Checkpoint written by
                :func:`~polyt5.training.save_checkpoint`.
            device: Device to load and run the reference model on.

        Returns:
            A :class:`ReferencePolicy` wrapping a freshly constructed model
            with the checkpoint's weights loaded.
        """
        payload = load_checkpoint(path, map_location=str(device))
        config = PolyT5Config.from_dict(payload["model_config"])
        model = PolyT5ForConditionalGeneration(config)
        model.load_state_dict(payload["model_state"])
        return cls(model, device=device)

    @torch.no_grad()
    def score(self, sequences: Tensor, prompt_ids: Tensor, prompt_mask: Tensor) -> Tensor:
        """Per-token log pi_ref of ``sequences``, via teacher forcing.

        Args:
            sequences: ``(n, gen_len)`` decoder-side token ids to score (e.g.
                :attr:`~polyt5.rl.rollout.RolloutBatch.sequences`) -- the
                decoder start token is NOT included; padded positions carry
                ``pad_token_id``.
            prompt_ids: ``(n, src_len)`` encoder input ids.
            prompt_mask: ``(n, src_len)`` encoder padding mask.

        Returns:
            ``(n, gen_len)`` log-probs of every position in ``sequences``
            under the reference policy's UNMODIFIED distribution (plain
            ``log_softmax`` of the logits from one teacher-forced forward
            pass -- there is no sampling here to filter). Values at trailing
            padding are meaningless (whatever the model assigns to
            ``pad_token_id`` given an all-padding decoder context) but
            harmless: this method is deliberately NOT responsible for
            masking, because it has no ground truth for which positions are
            real -- ``sequences == pad_token_id`` is not reliable (an early
            or high-temperature policy can legitimately sample the pad id as
            real content before EOS). Combine with the SAME per-token mask
            :func:`~polyt5.rl.rollout.sample_groups` produced
            (:attr:`~polyt5.rl.rollout.RolloutBatch.mask`), exactly as
            :func:`~polyt5.rl.grpo.grpo_loss`'s own ``mask`` argument already
            expects.
        """
        sequences = sequences.to(self.device)
        prompt_ids = prompt_ids.to(self.device)
        prompt_mask = prompt_mask.to(self.device)

        decoder_start = self.model.config.decoder_start_token_id
        start = torch.full(
            (sequences.shape[0], 1), decoder_start, dtype=sequences.dtype, device=self.device
        )
        decoder_input_ids = torch.cat([start, sequences[:, :-1]], dim=1)

        output = self.model(
            prompt_ids, attention_mask=prompt_mask, decoder_input_ids=decoder_input_ids
        )
        logprobs = torch.log_softmax(output.logits.float(), dim=-1)
        return logprobs.gather(2, sequences.unsqueeze(-1)).squeeze(-1)
