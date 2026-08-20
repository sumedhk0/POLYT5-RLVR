"""Autoregressive decoding: greedy and top-p / temperature sampling.

Paper context (Sahu et al., npj Artificial Intelligence 2026): the
CONDITIONAL-GENERATION experiments decode by sampling -- explicitly "instead
of beam search" -- with ``top_p`` in {0.75, 0.95}, temperature swept
0.1 -> 2.0 in increments of 0.1, 10,000 polymers per configuration and a
maximum output length of 200 tokens (hence :attr:`GenerationConfig.max_length`
defaults to 200). PROPERTY PREDICTION instead uses beam search with width 4;
see :mod:`polyt5.generation.beam_search`. Mixing the two regimes up is a
reproduction error.

Memory: only a ``(batch, vocab)`` slice of logits exists at any moment on the
KV-cache path -- the full ``(batch, length, vocab)`` tensor is never
materialised -- so long batched runs fit comfortably on a 12 GB GPU.

Gradients: :func:`generate` is wrapped in ``torch.no_grad()`` because
inference never needs a graph. The pieces a future RL rollout needs are kept
separate and pure: the logit processors in :mod:`polyt5.generation.sampling`,
and the per-token log-probs returned here (see :class:`GenerationOutput`).
Nothing RL-specific is implemented in this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from polyt5.generation.sampling import (
    apply_repetition_penalty,
    apply_temperature,
    top_k_filter,
    top_p_filter,
)

NEG_INF = float("-inf")

__all__ = ["GenerationConfig", "GenerationOutput", "batch_generate", "generate"]


@dataclass(kw_only=True)
class GenerationConfig:
    """Decoding hyperparameters for :func:`generate`.

    Attributes:
        max_length: Maximum number of tokens to GENERATE, excluding the
            decoder start token (which is never part of the returned
            sequences). The paper's conditional-generation runs use 200.
            # [AMBIGUITY] The paper says "maximum output length of 200 tokens"
            # without saying whether special tokens count; we count generated
            # tokens, excluding the decoder start token and including EOS.
        min_length: Minimum number of generated tokens; EOS is masked to
            ``-inf`` until a sequence is at least this long. ``1`` (the
            default) allows EOS immediately and is therefore a no-op.
        do_sample: Sample (True, the paper's conditional-generation regime) or
            decode greedily (False).
        temperature: Sampling temperature; the paper sweeps 0.1 -> 2.0.
            ``<= 0`` degenerates to greedy (see
            :func:`~polyt5.generation.sampling.apply_temperature`).
        top_p: Nucleus mass; the paper uses {0.75, 0.95}. ``1.0`` disables it.
        top_k: Optional top-k truncation. # [AMBIGUITY] Not stated in the
            paper; defaults to ``None`` (disabled), which is the
            paper-faithful setting.
        repetition_penalty: CTRL-style penalty. # [AMBIGUITY] Not stated in
            the paper; defaults to ``1.0`` (exact no-op).
        num_return_sequences: Samples to draw per input row. # [AMBIGUITY] The
            paper generates 10,000 polymers per configuration but does not say
            whether that comes from one sample per prompt or many; defaults to
            1. Rows are grouped per input: input ``i`` owns output rows
            ``i * num_return_sequences ... (i + 1) * num_return_sequences - 1``.
        eos_token_id: End-of-sequence id; a row that emits it is frozen.
        pad_token_id: Padding id used to fill frozen rows.
        decoder_start_token_id: First decoder input token (T5 uses the pad id).
        seed: Optional seed. # [AMBIGUITY] The paper reports no seeds. When
            set, a private ``torch.Generator`` is seeded with it, so sampling
            never touches -- and never perturbs -- the global RNG. An explicit
            ``generator=`` argument to :func:`generate` takes precedence.
        use_cache: Use incremental KV-cached decoding (True) or re-run the
            full decoder prefix each step (False). Results are identical; the
            uncached path exists to catch cache bugs and is much slower.
    """

    max_length: int = 200
    min_length: int = 1
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int | None = None
    repetition_penalty: float = 1.0
    num_return_sequences: int = 1
    eos_token_id: int
    pad_token_id: int
    decoder_start_token_id: int
    seed: int | None = None
    use_cache: bool = True

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If any field is out of range or mutually inconsistent.
        """
        if self.max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {self.max_length}.")
        if self.min_length < 0:
            raise ValueError(f"min_length must be >= 0, got {self.min_length}.")
        if self.max_length < self.min_length:
            raise ValueError(
                f"max_length ({self.max_length}) must be >= min_length ({self.min_length})."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}.")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be >= 1 or None, got {self.top_k}.")
        if self.repetition_penalty <= 0.0:
            raise ValueError(
                f"repetition_penalty must be > 0, got {self.repetition_penalty}."
            )
        if self.num_return_sequences < 1:
            raise ValueError(
                f"num_return_sequences must be >= 1, got {self.num_return_sequences}."
            )
        for name in ("eos_token_id", "pad_token_id", "decoder_start_token_id"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}.")


@dataclass
class GenerationOutput:
    """Result of :func:`generate` (and of
    :func:`~polyt5.generation.beam_search.beam_search`).

    Attributes:
        sequences: ``(batch * num_return_sequences, gen_len)`` generated token
            ids. The decoder start token is NOT included. Positions after a
            row's EOS are ``pad_token_id``. ``gen_len <= max_length``: the loop
            stops early once every row has finished.
        scores: ``(batch * num_return_sequences,)`` summed log-probability of
            each sequence -- the sum of :attr:`token_logprobs`, whose padded
            entries are exactly zero. For beam search this is the
            length-penalised score that decided the ranking.
        token_logprobs: ``(batch * num_return_sequences, gen_len)`` log-prob of
            each CHOSEN token.

            NOTE (important, and deliberate): these come from the UNMODIFIED
            model distribution -- ``log_softmax`` of the raw logits, BEFORE
            temperature scaling, top-p/top-k filtering, repetition penalty and
            min-length EOS masking. That makes them the true log-probs of the
            sampled tokens under the policy pi_theta, which is exactly what a
            policy-gradient / importance-sampling method needs; the modified
            distribution is only the proposal used for exploration. Padded
            positions of finished rows are set to ``0.0`` so that summing gives
            the sequence score.
        finished: ``(batch * num_return_sequences,)`` bool -- whether the row
            actually emitted EOS (False means it was truncated at
            ``max_length``).
        lengths: ``(batch * num_return_sequences,)`` number of generated
            tokens excluding padding, INCLUDING the terminal EOS when present.
    """

    sequences: Tensor
    scores: Tensor | None
    token_logprobs: Tensor | None
    finished: Tensor
    lengths: Tensor


def _resolve_generator(
    config: GenerationConfig, generator: torch.Generator | None, device: torch.device
) -> torch.Generator | None:
    """Pick the RNG for sampling.

    An explicitly passed ``generator`` always wins; otherwise ``config.seed``
    (if set) creates a private generator on ``device``. Sampling NEVER falls
    back to the global RNG when either is given, so generation is reproducible
    without disturbing (or being disturbed by) surrounding code.
    """
    if generator is not None:
        return generator
    if config.seed is None:
        return None
    return torch.Generator(device=device).manual_seed(config.seed)


def _next_token_logits(
    model: torch.nn.Module,
    *,
    encoder_hidden_states: Tensor,
    encoder_attention_mask: Tensor | None,
    decoder_input_ids: Tensor,
    step_input_ids: Tensor,
    past_key_values: object | None,
    use_cache: bool,
) -> tuple[Tensor, object | None]:
    """Compute ``(batch, vocab)`` logits for the next position.

    Args:
        model: The seq2seq model.
        encoder_hidden_states: Encoder output, already expanded to the
            decoding batch.
        encoder_attention_mask: Source padding mask, already expanded.
        decoder_input_ids: Full decoder prefix so far (uncached path).
        step_input_ids: Only the newest decoder token(s) (cached path).
        past_key_values: Cache from the previous step, or ``None``.
        use_cache: Whether to take the incremental path.

    Returns:
        Tuple of ``(logits, past_key_values)``; the cache is ``None`` on the
        uncached path.
    """
    if use_cache:
        logits, present = model.decode_step(
            step_input_ids,
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
        )
        return logits[:, -1, :], present
    outputs = model(
        None,
        attention_mask=encoder_attention_mask,
        decoder_input_ids=decoder_input_ids,
        encoder_outputs=encoder_hidden_states,
        use_cache=False,
    )
    return outputs.logits[:, -1, :], None


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
    *,
    config: GenerationConfig,
    generator: torch.Generator | None = None,
) -> GenerationOutput:
    """Autoregressively decode a batch with greedy or top-p/temperature sampling.

    Per-sequence stopping: as soon as a row emits ``eos_token_id`` it is
    frozen -- every later position of that row is ``pad_token_id``, its
    log-probs are ``0.0``, and its length stops growing -- while the other
    rows keep decoding. The loop exits when all rows are finished or after
    ``max_length`` tokens, whichever comes first.

    The module mode is NOT changed: call ``model.eval()`` first, or dropout
    will make even greedy decoding non-deterministic. (Leaving the mode alone
    is what lets a future RL rollout re-run the same code with dropout on.)

    Args:
        model: A :class:`~polyt5.model.PolyT5ForConditionalGeneration` (or any
            module exposing the same ``encode`` / ``decode_step`` / ``forward``
            contract).
        input_ids: ``(batch, src_len)`` source token ids.
        attention_mask: ``(batch, src_len)`` 1/0 source padding mask.
        config: Decoding hyperparameters.
        generator: Explicit RNG for sampling; takes precedence over
            ``config.seed``. Sampling never uses the global RNG when either is
            supplied.

    Returns:
        A :class:`GenerationOutput`. See its docstring for the exact
        distribution :attr:`GenerationOutput.token_logprobs` refers to.
    """
    device = input_ids.device
    rng = _resolve_generator(config, generator, device)
    num_return = config.num_return_sequences

    encoder_hidden_states = model.encode(input_ids, attention_mask=attention_mask)
    if num_return > 1:
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_return, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(num_return, dim=0)
    batch = encoder_hidden_states.shape[0]

    decoder_input_ids = torch.full(
        (batch, 1), config.decoder_start_token_id, dtype=torch.long, device=device
    )
    step_input_ids = decoder_input_ids
    past_key_values: object | None = None

    unfinished = torch.ones(batch, dtype=torch.bool, device=device)
    lengths = torch.zeros(batch, dtype=torch.long, device=device)
    tokens: list[Tensor] = []
    logprobs: list[Tensor] = []

    for step in range(config.max_length):
        logits, past_key_values = _next_token_logits(
            model,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            step_input_ids=step_input_ids,
            past_key_values=past_key_values,
            use_cache=config.use_cache,
        )
        logits = logits.float()

        # Log-probs of the UNMODIFIED policy, captured before any processing.
        raw_logprobs = torch.log_softmax(logits, dim=-1)

        processed = logits
        if config.repetition_penalty != 1.0 and tokens:
            generated = torch.stack(tokens, dim=1)
            processed = apply_repetition_penalty(processed, generated, config.repetition_penalty)
        if step + 1 < config.min_length:
            # Emitting EOS now would yield a sequence shorter than min_length.
            processed = processed.clone()
            processed[:, config.eos_token_id] = NEG_INF

        if config.do_sample:
            processed = apply_temperature(processed, config.temperature)
            if config.top_k is not None:
                processed = top_k_filter(processed, config.top_k)
            processed = top_p_filter(processed, config.top_p)
            probs = torch.softmax(processed, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1, generator=rng).squeeze(1)
        else:
            next_token = processed.argmax(dim=-1)

        token_logprob = raw_logprobs.gather(1, next_token[:, None]).squeeze(1)

        # Freeze rows that already finished.
        pad = torch.full_like(next_token, config.pad_token_id)
        next_token = torch.where(unfinished, next_token, pad)
        token_logprob = torch.where(unfinished, token_logprob, torch.zeros_like(token_logprob))
        lengths = lengths + unfinished.long()

        tokens.append(next_token)
        logprobs.append(token_logprob)

        unfinished = unfinished & (next_token != config.eos_token_id)
        if not bool(unfinished.any()):
            break

        step_input_ids = next_token[:, None]
        decoder_input_ids = torch.cat([decoder_input_ids, step_input_ids], dim=1)

    sequences = torch.stack(tokens, dim=1)
    token_logprobs = torch.stack(logprobs, dim=1)
    return GenerationOutput(
        sequences=sequences,
        scores=token_logprobs.sum(dim=1),
        token_logprobs=token_logprobs,
        finished=~unfinished,
        lengths=lengths,
    )


def batch_generate(
    model: torch.nn.Module,
    list_of_input_id_lists: Sequence[Sequence[int]],
    *,
    config: GenerationConfig,
    pad_id: int,
    device: torch.device,
    batch_size: int = 32,
) -> list[list[int]]:
    """Chunk, pad, generate and strip -- the convenience wrapper for scripts.

    Sources are padded to the longest row WITHIN each chunk (not globally), so
    memory scales with the chunk rather than the corpus. Outputs are stripped
    of padding and truncated at the first EOS, which is itself dropped: the
    returned lists contain content tokens only and are ready for the
    tokenizer's ``decode``.

    Args:
        model: Model in eval mode.
        list_of_input_id_lists: One list of source ids per example.
        config: Decoding hyperparameters (shared by every chunk, so a fixed
            ``config.seed`` gives every chunk the same RNG stream -- pass an
            explicit generator to :func:`generate` instead if that matters).
        pad_id: Padding id for the SOURCE side (usually the same as
            ``config.pad_token_id``).
        device: Device to run on.
        batch_size: Number of source rows per chunk.

    Returns:
        ``len(list_of_input_id_lists) * config.num_return_sequences`` lists of
        token ids, grouped per input (input ``i`` first, then its extra
        samples).

    Raises:
        ValueError: If ``batch_size < 1`` or an input row is empty.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

    results: list[list[int]] = []
    for start in range(0, len(list_of_input_id_lists), batch_size):
        chunk = list_of_input_id_lists[start : start + batch_size]
        if any(len(row) == 0 for row in chunk):
            raise ValueError("empty input sequence in batch_generate.")
        width = max(len(row) for row in chunk)

        input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(chunk), width), dtype=torch.long, device=device)
        for i, row in enumerate(chunk):
            input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
            attention_mask[i, : len(row)] = 1

        output = generate(model, input_ids, attention_mask, config=config)
        for row in output.sequences.tolist():
            trimmed: list[int] = []
            for token in row:
                if token == config.eos_token_id:
                    break
                if token == config.pad_token_id:
                    continue
                trimmed.append(token)
            results.append(trimmed)
    return results
