"""Cycle consistency: generate for a target Tg, then score what you generated.

**This ships behind a flag and the flag defaults to OFF.**

The signal is circular on its own. A model can satisfy it by being
*consistently wrong*: generate something odd, confidently mispredict it as
500 K, incur zero loss. Anchored by 7,367 real labels it is a legitimate
semi-supervised regulariser -- back-translation is anchored by real parallel
text in exactly the same way -- but as a primary objective it measures nothing.
So:

* :attr:`CycleConfig.enabled` is ``False`` by default, and
  :func:`build_cycle_loss` returns ``None`` when it is.
* No arm in :mod:`polyt5.training.group_a` turns it on, and a test pins that.
* :class:`polyt5.training.multitask_trainer.GroupATrainer` refuses a cycle loss
  for an arm whose flag is off, and refuses an arm whose flag is on without one.

Sampling happens under ``no_grad`` -- the gradient flows through the RESCORING
pass, not through the sampling, which is not differentiable anyway.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from polyt5.data.prepare import format_property_value
from polyt5.generation import GenerationConfig, generate
from polyt5.model.heads import weighted_huber_loss
from polyt5.model.multitask import PolyT5MultiTask

__all__ = ["CycleConfig", "build_cycle_loss"]


@dataclass(frozen=True)
class CycleConfig:
    """Sampling and weighting knobs for the cycle term.

    Attributes:
        enabled: Off by default. See the module docstring for why.
        max_length: Decode cap for the generated PSELFIES.
        temperature: Sampling temperature; must be positive.
        top_p: Nucleus cutoff; must be in ``(0, 1]``.
        seed: Sampling seed, so a cycle loss is reproducible.
        huber_delta: Huber transition point, in standardised units.
    """

    enabled: bool = False
    max_length: int = 200
    temperature: float = 1.0
    top_p: float = 0.95
    seed: int = 0
    huber_delta: float = 1.0


def build_cycle_loss(
    model: PolyT5MultiTask,
    tokenizer,
    *,
    config: CycleConfig,
    device: str,
) -> Callable[[Tensor], Tensor] | None:
    """Build the cycle-consistency loss callable, or ``None`` when disabled.

    Args:
        model: The multi-task model; needs a regression head to score with.
        tokenizer: The tokenizer the model was trained with.
        config: Sampling and weighting knobs.
        device: Torch device string.

    Returns:
        A callable taking ``(batch,)`` STANDARDISED Tg targets and returning a
        scalar loss, or ``None`` when ``config.enabled`` is ``False``.

    Raises:
        ValueError: If enabled on a model with no regression head, or with a
            non-positive temperature or an out-of-range ``top_p``.
    """
    if not config.enabled:
        return None
    if model.tg_head is None:
        raise ValueError(
            "cycle consistency needs a regression head to score its own generations with; "
            "this model has none, so there is nothing to close the loop"
        )
    if config.temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {config.temperature}")
    if not 0.0 < config.top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {config.top_p}")

    def cycle_loss(standardised_targets: Tensor) -> Tensor:
        if standardised_targets.numel() == 0:
            return torch.zeros((), device=device)

        kelvin = standardised_targets.detach() * model.tg_std + model.tg_mean
        prompts = [format_property_value(float(value)) for value in kelvin]
        encoded = tokenizer.batch_encode(
            prompts, add_eos=True, max_length=config.max_length,
            padding=True, truncation=True,
        )
        prompt_ids = torch.tensor(encoded["input_ids"], device=device)
        prompt_mask = torch.tensor(encoded["attention_mask"], device=device)

        with torch.no_grad():
            sampled = generate(
                model.backbone, prompt_ids, prompt_mask,
                config=GenerationConfig(
                    max_length=config.max_length,
                    do_sample=True,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    eos_token_id=tokenizer.eos_id,
                    pad_token_id=tokenizer.pad_id,
                    decoder_start_token_id=tokenizer.decoder_start_token_id,
                    seed=config.seed,
                ),
            )
        texts = tokenizer.batch_decode(sampled.sequences.tolist(), skip_special_tokens=True)
        rescored = tokenizer.batch_encode(
            texts, add_eos=True, max_length=config.max_length,
            padding=True, truncation=True,
        )
        # The RESCORING pass carries the gradient; sampling above did not.
        output = model.forward_regression(
            torch.tensor(rescored["input_ids"], device=device),
            torch.tensor(rescored["attention_mask"], device=device),
        )
        assert output.tg_standardised is not None
        return weighted_huber_loss(
            output.tg_standardised,
            standardised_targets.to(output.tg_standardised.dtype),
            delta=config.huber_delta,
        )

    return cycle_loss
