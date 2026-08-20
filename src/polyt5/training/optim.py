"""Optimizer and LR-schedule construction for polyT5 training.

The paper (Sahu et al., npj AI 2026) specifies AdamW for both pretraining and
fine-tuning, and LR 3e-4 / weight decay 0.01 for fine-tuning. It does NOT
specify a pretraining learning rate, any LR schedule, warmup, or which
parameters receive weight decay — those choices are ours and are marked
``[AMBIGUITY]`` here and in :mod:`polyt5.training.trainer`.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Build AdamW with weight decay applied ONLY to weight matrices.

    [AMBIGUITY] The paper says "AdamW" and (for fine-tuning) "weight decay
    0.01" but never says which parameters are decayed. We follow standard
    transformer practice (BERT/T5/GPT reference implementations): decay 2-D+
    weight matrices; exclude biases, embeddings, and all LayerNorm /
    T5LayerNorm parameters. Decaying RMS-norm scales or embeddings is known
    to hurt, so this is the conservative default.

    Args:
        model: The model whose trainable parameters are optimized. Tied
            parameters (e.g. lm_head sharing the embedding) are counted once
            because ``named_parameters()`` deduplicates shared tensors.
        lr: Learning rate.
        weight_decay: Decoupled weight decay for the decay group.
        betas: Adam betas. [AMBIGUITY] Not stated in the paper; torch default.
        eps: Adam epsilon. [AMBIGUITY] Not stated in the paper; torch default.

    Returns:
        A :class:`torch.optim.AdamW` with exactly two param groups:
        ``[decayed weight matrices, everything else at weight_decay=0]``.
    """
    # Parameters owned by embedding or any *LayerNorm module are never decayed,
    # even when 2-D (embeddings are 2-D matrices but are lookup tables, not
    # projections).
    no_decay_names: set[str] = set()
    for module_name, module in model.named_modules():
        is_norm = isinstance(module, nn.LayerNorm) or "layernorm" in type(module).__name__.lower()
        if isinstance(module, nn.Embedding) or is_norm:
            for param_name, _ in module.named_parameters(recurse=False):
                full = f"{module_name}.{param_name}" if module_name else param_name
                no_decay_names.add(full)

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in no_decay_names or name.endswith(".bias") or param.ndim < 2:
            no_decay.append(param)
        else:
            decay.append(param)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    name: str | None,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    **kwargs,
) -> LambdaLR | None:
    """Build an LR schedule as a multiplicative factor on the base LR.

    [AMBIGUITY] The paper states NO schedule at all (for either stage). The
    trainer's default is ``"inverse_sqrt"`` — the schedule the original T5
    paper used for pretraining — chosen because polyT5 *is* a T5; see
    :class:`polyt5.training.trainer.TrainerConfig`.

    Supported names (step counts are OPTIMIZER steps, not micro-batches):
        - ``"none"`` / ``None``: no scheduler (constant base LR), returns None.
        - ``"constant"``: linear warmup, then flat at the base LR.
        - ``"linear"``: linear warmup, then linear decay to 0 at
          ``num_training_steps``.
        - ``"inverse_sqrt"``: linear warmup, then
          ``lr = base * sqrt(timescale / step)`` where ``timescale`` defaults
          to ``num_warmup_steps`` (or the ``timescale`` kwarg, or 10000 when
          there is no warmup) — the original T5 pretraining shape.
        - ``"cosine"``: linear warmup, then cosine decay to 0 at
          ``num_training_steps`` (``num_cycles`` kwarg, default 0.5).

    Returns:
        A :class:`torch.optim.lr_scheduler.LambdaLR`, or ``None``.
    """
    if name is None or name == "none":
        return None

    warmup = max(0, int(num_warmup_steps))

    def warmup_factor(step: int) -> float:
        return step / max(1, warmup)

    if name == "constant":

        def factor(step: int) -> float:
            return warmup_factor(step) if step < warmup else 1.0

    elif name == "linear":

        def factor(step: int) -> float:
            if step < warmup:
                return warmup_factor(step)
            remaining = num_training_steps - step
            return max(0.0, remaining / max(1, num_training_steps - warmup))

    elif name == "inverse_sqrt":
        timescale = kwargs.get("timescale") or warmup or 10000

        def factor(step: int) -> float:
            if step < warmup:
                return warmup_factor(step)
            return math.sqrt(timescale / max(step, timescale))

    elif name == "cosine":
        num_cycles = kwargs.get("num_cycles", 0.5)

        def factor(step: int) -> float:
            if step < warmup:
                return warmup_factor(step)
            progress = (step - warmup) / max(1, num_training_steps - warmup)
            progress = min(1.0, progress)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress)))

    else:
        raise ValueError(
            f"Unknown scheduler {name!r}; expected one of "
            "'none', 'constant', 'linear', 'inverse_sqrt', 'cosine'"
        )

    return LambdaLR(optimizer, lr_lambda=factor)
