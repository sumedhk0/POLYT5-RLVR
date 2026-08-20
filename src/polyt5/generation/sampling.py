"""Pure logit-processing primitives for sampling-based decoding.

Paper context (Sahu et al., npj Artificial Intelligence 2026): the
conditional-generation experiments decode by SAMPLING -- explicitly "instead
of beam search" -- sweeping ``top_p`` over {0.75, 0.95} and temperature from
0.1 to 2.0 in increments of 0.1, generating 10,000 polymers per configuration
with a maximum output length of 200 tokens. The reported best (epochs,
temperature, top_p) settings are small (14, 0.9, 0.75), medium (6, 1.1, 0.75)
and large (8, 1.1, 0.95). The paper also reports the expected trade-off:
higher ``top_p`` gives more diversity but more invalid structures, lower
``top_p`` fewer invalid structures but more duplicates.

Design contract for every function in this module:

* **Pure.** The input tensor is never mutated; a new tensor is returned.
* **Shape.** Operates on ``(batch, vocab)`` logits, row-independently.
* **No RNG.** Nothing here samples, so nothing here touches a random seed.
* **Autograd-friendly.** No in-place ops on the input and no
  ``torch.no_grad()``, so a future RL rollout can re-run these inside a graph
  that requires gradients (this module implements nothing RL-specific).

Masked-out tokens are set to ``-inf`` so that a subsequent ``softmax`` gives
them exactly zero probability.
"""

from __future__ import annotations

import torch
from torch import Tensor

NEG_INF = float("-inf")

__all__ = [
    "apply_repetition_penalty",
    "apply_temperature",
    "top_k_filter",
    "top_p_filter",
]


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Rescale logits by a sampling temperature.

    ``T < 1`` sharpens the distribution (lower entropy), ``T > 1`` flattens it
    (higher entropy), and ``T == 1`` is the identity (division by 1.0 is exact
    in IEEE floating point, so the returned values compare equal).

    Greedy signalling: ``temperature <= 0`` is a degenerate temperature and is
    NOT applied by division (that would divide by zero). Instead it returns a
    "greedy one-hot" logit vector -- ``0.0`` at each row's argmax and ``-inf``
    everywhere else -- whose softmax puts all probability on the argmax token.
    Sampling from that distribution is exactly greedy decoding, so callers do
    not need a separate code path.

    Args:
        logits: ``(batch, vocab)`` raw logits.
        temperature: Sampling temperature. The paper sweeps 0.1 -> 2.0.

    Returns:
        A new ``(batch, vocab)`` tensor of rescaled logits.
    """
    if temperature <= 0.0:
        greedy = torch.full_like(logits, NEG_INF)
        argmax = logits.argmax(dim=-1, keepdim=True)
        return greedy.scatter(-1, argmax, 0.0)
    return logits / temperature


def top_p_filter(logits: Tensor, top_p: float, *, min_tokens_to_keep: int = 1) -> Tensor:
    """Nucleus (top-p) filtering.

    Keeps the smallest set of highest-probability tokens whose cumulative
    softmax probability reaches ``top_p`` and sets every other logit to
    ``-inf``.

    Off-by-one convention (the classic bug): the token that CROSSES the
    threshold IS KEPT. Sorting descending with cumulative probabilities
    ``c_0 <= c_1 <= ...``, token ``i`` is removed iff ``c_{i-1} > top_p``;
    the top token is never removed. So for probabilities
    ``[0.6, 0.3, 0.07, 0.03]`` and ``top_p = 0.75`` the surviving set is
    ``{0, 1}`` -- token 1 is what makes the cumulative mass reach 0.75, and
    dropping it would leave a nucleus that never reaches ``top_p`` at all.

    Args:
        logits: ``(batch, vocab)`` raw logits.
        top_p: Nucleus probability mass in ``(0, 1]``. The paper uses
            {0.75, 0.95}. ``top_p == 1.0`` is a no-op.
        min_tokens_to_keep: Never filter below this many tokens per row. With
            the default of 1 this is already implied by the convention above
            (the argmax token always survives, even when its own probability
            exceeds ``top_p``); values > 1 widen the floor.

    Returns:
        A new ``(batch, vocab)`` tensor with filtered positions set to
        ``-inf``.

    Raises:
        ValueError: If ``top_p`` is outside ``(0, 1]`` or
            ``min_tokens_to_keep < 1``.
    """
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}.")
    if min_tokens_to_keep < 1:
        raise ValueError(f"min_tokens_to_keep must be >= 1, got {min_tokens_to_keep}.")
    if top_p == 1.0:
        return logits.clone()

    sorted_logits, sorted_index = torch.sort(logits, dim=-1, descending=True)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

    sorted_remove = cumulative > top_p
    # Shift right so the crossing token is kept, and never drop the argmax.
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False
    if min_tokens_to_keep > 1:
        sorted_remove[..., : min(min_tokens_to_keep, sorted_remove.shape[-1])] = False

    remove = torch.zeros_like(sorted_remove).scatter(-1, sorted_index, sorted_remove)
    return logits.masked_fill(remove, NEG_INF)


def top_k_filter(logits: Tensor, top_k: int) -> Tensor:
    """Keep only the ``top_k`` highest logits per row, masking the rest.

    # [AMBIGUITY] The paper never mentions top-k sampling; it reports only
    # top-p and temperature. This is provided for completeness and defaults to
    # OFF (``GenerationConfig.top_k = None``). Do not enable it when
    # reproducing the paper's numbers.

    Args:
        logits: ``(batch, vocab)`` raw logits.
        top_k: Number of tokens to keep. Values larger than the vocabulary are
            clamped, which makes the call a no-op.

    Returns:
        A new ``(batch, vocab)`` tensor with filtered positions set to
        ``-inf``.

    Raises:
        ValueError: If ``top_k < 1``.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}.")
    k = min(top_k, logits.shape[-1])
    if k == logits.shape[-1]:
        return logits.clone()
    kth_value = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth_value, NEG_INF)


def apply_repetition_penalty(logits: Tensor, generated: Tensor, penalty: float) -> Tensor:
    """Apply the CTRL-style repetition penalty to already-generated tokens.

    Following Keskar et al. (CTRL, 2019): a positive logit for a previously
    generated token is DIVIDED by ``penalty`` and a negative one is MULTIPLIED
    by it, so ``penalty > 1`` always makes repetition less likely. The penalty
    is applied once per distinct token id regardless of how often it occurs.

    # [AMBIGUITY] The paper does not mention a repetition penalty. The default
    # (``GenerationConfig.repetition_penalty = 1.0``) is an exact no-op, which
    # is the paper-faithful setting.

    Args:
        logits: ``(batch, vocab)`` raw logits.
        generated: ``(batch, gen_len)`` token ids produced so far. The caller
            decides what belongs here -- padding ids inside finished rows are
            penalised like any other token, which is harmless because those
            rows are frozen.
        penalty: Penalty factor > 0. ``1.0`` is the identity.

    Returns:
        A new ``(batch, vocab)`` tensor of penalised logits.

    Raises:
        ValueError: If ``penalty <= 0`` or the batch dimensions disagree.
    """
    if penalty <= 0.0:
        raise ValueError(f"repetition penalty must be > 0, got {penalty}.")
    if penalty == 1.0 or generated.numel() == 0:
        return logits.clone()
    if generated.shape[0] != logits.shape[0]:
        raise ValueError(
            f"batch mismatch: logits {logits.shape[0]} vs generated {generated.shape[0]}."
        )

    gathered = logits.gather(-1, generated)
    penalised = torch.where(gathered < 0, gathered * penalty, gathered / penalty)
    return logits.scatter(-1, generated, penalised)
