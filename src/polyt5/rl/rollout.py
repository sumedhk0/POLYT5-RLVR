"""Group rollout sampling: the candidates GRPO computes advantages over.

For each target Tg value, :func:`sample_groups` draws a GROUP of candidate
polymers from the current policy, keeping the per-token log-probabilities
under the UNMODIFIED policy distribution -- exactly the ``pi_theta_old``
:func:`~polyt5.rl.grpo.grpo_loss` needs for its importance ratio.

Two things this module is careful about:

1. Contiguous group layout. :func:`~polyt5.rl.advantages.group_advantages`
   reshapes a flat reward array into ``(-1, group_size)`` and normalises
   within rows, so a prompt's group members must occupy consecutive
   positions. Interleaving prompts instead would compute advantages across
   unrelated prompts and silently corrupt the whole algorithm.
2. Rollout batch size. Measured on this hardware, generating in batches of
   128 gives ~125 candidates/second; batching 512 at once drops to
   ~36/second -- nearly 4x worse. Generation is therefore always chunked at
   :data:`ROLLOUT_CHUNK_SIZE`, regardless of how many candidates the caller
   asks for.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from polyt5.data.prepare import format_property_value
from polyt5.generation import GenerationConfig, generate
from polyt5.tokenization import PolyT5Tokenizer

__all__ = ["ROLLOUT_CHUNK_SIZE", "RolloutBatch", "sample_groups"]

#: Candidates per `generate()` call. See the module docstring -- this is a
#: measured hardware optimum, not a tunable knob.
ROLLOUT_CHUNK_SIZE = 128


@dataclass(frozen=True)
class RolloutBatch:
    """One step's worth of sampled candidates, grouped per prompt.

    Every field's row ``i`` corresponds to the same candidate; group members
    of one prompt occupy ``group_size`` CONSECUTIVE rows (see the module
    docstring), which is what lets :func:`~polyt5.rl.advantages.group_advantages`
    reshape directly.

    Attributes:
        sequences: ``(n, gen_len)`` generated token ids (decoder side; the
            decoder start token is not included). Positions after a row's
            EOS are ``pad_token_id``.
        logprobs: ``(n, gen_len)`` log-prob of each chosen token under the
            UNMODIFIED policy distribution (before temperature / top-p /
            top-k filtering) -- the correct ``pi_theta_old``. Padded
            positions are ``0.0``.
        mask: ``(n, gen_len)`` 1 on real tokens, 0 on padding -- built from
            each row's generated length (not from ``sequences != pad_token_id``,
            which a policy can legitimately violate; see :func:`sample_groups`).
        prompt_ids: ``(n, src_len)`` encoder input ids, one row per candidate
            (a prompt's encoding is repeated across its group).
        prompt_mask: ``(n, src_len)`` encoder padding mask, matching
            ``prompt_ids``.
        targets: The Tg value each row's prompt was formatted from, length
            ``n``, in the same contiguous group order as every tensor field.
        texts: Decoded candidate strings (special tokens stripped), length
            ``n``.
    """

    sequences: Tensor
    logprobs: Tensor
    mask: Tensor
    prompt_ids: Tensor
    prompt_mask: Tensor
    targets: list[float]
    texts: list[str]


def _pad_and_cat(chunks: list[Tensor], *, pad_value: float) -> Tensor:
    """Right-pad ``(rows, len)`` chunks to a common width, then stack rows.

    Each chunk comes from an independent :func:`generate` call, so its
    ``gen_len`` can differ (that call's own rows may all have finished
    earlier). Padding matches the convention already used for finished rows
    inside :func:`generate`: pad-token ids / zero log-probs.

    Args:
        chunks: Per-chunk tensors, each ``(chunk_rows, chunk_len)``.
        pad_value: Fill value for the padded tail.

    Returns:
        ``(sum(chunk_rows), max(chunk_len))`` tensor.
    """
    if len(chunks) == 1:
        return chunks[0]
    width = max(chunk.shape[1] for chunk in chunks)
    padded = []
    for chunk in chunks:
        gap = width - chunk.shape[1]
        if gap:
            filler = torch.full(
                (chunk.shape[0], gap), pad_value, dtype=chunk.dtype, device=chunk.device
            )
            chunk = torch.cat([chunk, filler], dim=1)
        padded.append(chunk)
    return torch.cat(padded, dim=0)


def sample_groups(
    model: torch.nn.Module,
    tokenizer: PolyT5Tokenizer,
    *,
    targets: Sequence[float],
    group_size: int,
    max_length: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
    device: str | torch.device = "cpu",
) -> RolloutBatch:
    """Sample a contiguous group of candidates per target Tg value.

    Args:
        model: Policy model (a
            :class:`~polyt5.model.PolyT5ForConditionalGeneration` or anything
            exposing the same ``encode`` / ``decode_step`` contract). Its
            mode (train/eval) is not changed here -- callers decide whether
            dropout is on during rollout.
        tokenizer: Tokenizer used to format and encode prompts and decode
            candidates.
        targets: Tg values to condition on, one prompt per value.
        group_size: Candidates sampled per prompt (>= 1).
        max_length: Maximum number of tokens to generate per candidate.
        temperature: Sampling temperature (paper sweeps 0.1 -> 2.0).
        top_p: Nucleus mass (paper uses 0.75 or 0.95).
        top_k: Optional top-k truncation.
        repetition_penalty: CTRL-style repetition penalty.
        seed: Optional seed. When set, one private ``torch.Generator`` is
            created and reused across every chunked :func:`generate` call, so
            the RNG stream is continuous across chunk boundaries rather than
            restarting per chunk, while two calls with the same seed still
            reproduce bit-identical output.
        device: Device to encode prompts on and run generation on.

    Returns:
        A :class:`RolloutBatch` with ``group_size * len(targets)`` rows,
        grouped contiguously per prompt in the order ``targets`` was given.

    Raises:
        ValueError: If ``targets`` is empty or ``group_size < 1``.
    """
    if not targets:
        raise ValueError("targets must not be empty.")
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}.")

    device_t = torch.device(device)

    prompt_texts = [format_property_value(value) for value in targets]
    expanded_texts = [text for text in prompt_texts for _ in range(group_size)]
    expanded_targets = [value for value in targets for _ in range(group_size)]

    encoded = tokenizer.batch_encode(expanded_texts, add_eos=True, padding=True)
    prompt_ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=device_t)
    prompt_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long, device=device_t)

    config = GenerationConfig(
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_id,
        pad_token_id=tokenizer.pad_id,
        decoder_start_token_id=tokenizer.decoder_start_token_id,
        seed=seed,
    )
    # An explicit generator takes precedence over config.seed inside generate()
    # (see its docstring) and, unlike config.seed alone, keeps advancing across
    # chunks instead of restarting every 128-row call.
    generator = torch.Generator(device=device_t).manual_seed(seed) if seed is not None else None

    n = prompt_ids.shape[0]
    sequence_chunks: list[Tensor] = []
    logprob_chunks: list[Tensor] = []
    length_chunks: list[Tensor] = []
    for start in range(0, n, ROLLOUT_CHUNK_SIZE):
        end = start + ROLLOUT_CHUNK_SIZE
        output = generate(
            model,
            prompt_ids[start:end],
            prompt_mask[start:end],
            config=config,
            generator=generator,
        )
        sequence_chunks.append(output.sequences)
        # token_logprobs are under the UNMODIFIED distribution (see generate()'s
        # docstring) -- exactly pi_theta_old for GRPO's importance ratio. Do not
        # recompute from the filtered/processed logits.
        logprob_chunks.append(output.token_logprobs)
        length_chunks.append(output.lengths)

    sequences = _pad_and_cat(sequence_chunks, pad_value=tokenizer.pad_id)
    logprobs = _pad_and_cat(logprob_chunks, pad_value=0.0)
    lengths = torch.cat(length_chunks, dim=0)
    # NOTE: mask is built from `lengths` (generate()'s own `unfinished`
    # bookkeeping), NOT from `sequences != pad_id`. pad_token_id is an
    # ordinary, embeddable vocabulary entry -- an early-training or
    # high-temperature policy can legitimately sample it as real content
    # before emitting EOS, which a naive `!= pad_id` mask would misclassify
    # as padding and silently drop from the loss.
    positions = torch.arange(sequences.shape[1], device=sequences.device)
    mask = (positions[None, :] < lengths[:, None]).long()
    texts = tokenizer.batch_decode(sequences.tolist(), skip_special_tokens=True)

    return RolloutBatch(
        sequences=sequences,
        logprobs=logprobs,
        mask=mask,
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        targets=expanded_targets,
        texts=texts,
    )
