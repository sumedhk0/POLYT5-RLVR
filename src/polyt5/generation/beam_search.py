"""Beam search decoding -- the paper's PROPERTY-PREDICTION regime.

Paper context (Sahu et al., npj Artificial Intelligence 2026): property
prediction decodes with beam search, beam width 4, and the decoded strings are
then "decoded into floating-point numbers, and filtered to remove any invalid
or non-numeric outputs" (that numeric parsing/filtering belongs to the
evaluation track, not here). Conditional generation uses sampling instead --
see :mod:`polyt5.generation.generate`.

The search is deterministic by construction: no RNG is involved anywhere.

Memory: as in :mod:`polyt5.generation.generate`, only a
``(batch * num_beams, vocab)`` logit slice exists per step; the KV cache is
reordered in place of re-running the prefix.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from polyt5.generation.generate import GenerationOutput

NEG_INF = float("-inf")

__all__ = ["BeamSearchConfig", "beam_search", "length_penalized_score"]


@dataclass(kw_only=True)
class BeamSearchConfig:
    """Hyperparameters for :func:`beam_search`.

    Attributes:
        num_beams: Beam width. The paper uses 4 for property prediction.
        max_length: Maximum number of tokens to generate, excluding the
            decoder start token.
        length_penalty: Exponent in :func:`length_penalized_score`. ``1.0``
            normalises by the raw length (plain mean log-prob), ``0.0`` ranks
            by the summed log-prob, ``> 1`` favours longer sequences.
            # [AMBIGUITY] The paper states the beam width but not the length
            # penalty; 1.0 is the HuggingFace default and is what we use.
        early_stopping: Stop a batch row as soon as ``num_beams`` finished
            hypotheses exist. # [AMBIGUITY] Not stated in the paper; True is
            the common default and only affects how much extra search happens
            after the beam is already full.
        eos_token_id: End-of-sequence id; terminates a hypothesis.
        pad_token_id: Padding id for the returned (ragged) sequences.
        decoder_start_token_id: First decoder input token (T5 uses the pad id).

    Raises:
        ValueError: If ``num_beams < 1`` or ``max_length < 1``.
    """

    num_beams: int = 4
    max_length: int = 200
    length_penalty: float = 1.0
    early_stopping: bool = True
    eos_token_id: int
    pad_token_id: int
    decoder_start_token_id: int

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.num_beams < 1:
            raise ValueError(f"num_beams must be >= 1, got {self.num_beams}.")
        if self.max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {self.max_length}.")
        for name in ("eos_token_id", "pad_token_id", "decoder_start_token_id"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}.")


def length_penalized_score(sum_logprob: float, length: int, length_penalty: float) -> float:
    """Length-normalised beam score ``sum_logprob / length ** length_penalty``.

    Because ``sum_logprob`` is negative, a LARGER ``length_penalty`` shrinks
    the magnitude of long sequences' scores and therefore favours them, while
    ``length_penalty == 0`` reduces to the raw summed log-probability and
    favours short sequences. (This is the HuggingFace / Wu et al. 2016
    convention.)

    Args:
        sum_logprob: Total log-probability of the hypothesis.
        length: Number of generated tokens, including the terminal EOS.
        length_penalty: Normalisation exponent.

    Returns:
        The ranking score (higher is better).
    """
    return sum_logprob / (max(length, 1) ** length_penalty)


class _BeamHypotheses:
    """Fixed-size pool of the best finished hypotheses for one batch row."""

    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool) -> None:
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.beams: list[tuple[float, Tensor, Tensor]] = []
        self.worst_score = float("inf")

    def __len__(self) -> int:
        return len(self.beams)

    def add(self, tokens: Tensor, token_logprobs: Tensor, sum_logprob: float) -> None:
        """Insert a finished hypothesis, evicting the worst one when full."""
        score = length_penalized_score(sum_logprob, int(tokens.numel()), self.length_penalty)
        if len(self.beams) < self.num_beams or score > self.worst_score:
            self.beams.append((score, tokens, token_logprobs))
            if len(self.beams) > self.num_beams:
                worst_index = min(range(len(self.beams)), key=lambda i: self.beams[i][0])
                del self.beams[worst_index]
            self.worst_score = min(b[0] for b in self.beams)

    def is_done(self, best_running_sum: float, current_length: int) -> bool:
        """Whether this row can stop expanding.

        Args:
            best_running_sum: Best summed log-prob still on the beam.
            current_length: Number of tokens generated so far.

        Returns:
            True when the pool is full and either ``early_stopping`` is set or
            no running beam could ever beat the worst stored hypothesis.
        """
        if len(self.beams) < self.num_beams:
            return False
        if self.early_stopping:
            return True
        best_possible = length_penalized_score(
            best_running_sum, current_length, self.length_penalty
        )
        return self.worst_score >= best_possible


def _reorder_cache(past_key_values: tuple, index: Tensor) -> tuple:
    """Select cache rows for the surviving beams.

    The model's cache is a per-layer ``((self_k, self_v), (cross_k, cross_v))``
    tuple with the batch on dim 0; every tensor is gathered with the same
    index.
    """
    return tuple(
        tuple(tuple(tensor.index_select(0, index) for tensor in group) for group in layer)
        for layer in past_key_values
    )


@torch.no_grad()
def beam_search(
    model: torch.nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
    *,
    config: BeamSearchConfig,
) -> GenerationOutput:
    """Standard beam search with length-normalised scores.

    At each step every live beam is expanded, the ``2 * num_beams`` best
    continuations per batch row are considered, EOS continuations are retired
    into that row's hypothesis pool (a finished beam stops expanding but stays
    in the pool and can still win), and the best ``num_beams`` non-EOS
    continuations become the next beam. Rows finish independently.

    The module mode is NOT changed: call ``model.eval()`` first, or dropout
    will break determinism.

    Args:
        model: A :class:`~polyt5.model.PolyT5ForConditionalGeneration` (or any
            module with the same ``encode`` / ``decode_step`` contract).
        input_ids: ``(batch, src_len)`` source token ids.
        attention_mask: ``(batch, src_len)`` 1/0 source padding mask.
        config: Beam search hyperparameters.

    Returns:
        A :class:`~polyt5.generation.generate.GenerationOutput` with one row
        per input (the single best hypothesis).
        # [AMBIGUITY] The paper does not say how many beams are returned; we
        # return the top-1 hypothesis only, which is what its "decoded into
        # floating-point numbers" property-prediction pipeline consumes.
        :attr:`~polyt5.generation.generate.GenerationOutput.scores` holds the
        LENGTH-PENALISED score that decided the ranking -- unlike
        :func:`~polyt5.generation.generate.generate`, where it is the plain
        sum of the token log-probs. ``token_logprobs`` are the log-probs of
        the chosen tokens under the unmodified model distribution, padded with
        ``0.0``.
    """
    device = input_ids.device
    batch = input_ids.shape[0]
    beams = config.num_beams
    rows = batch * beams

    encoder_hidden_states = model.encode(input_ids, attention_mask=attention_mask)
    encoder_hidden_states = encoder_hidden_states.repeat_interleave(beams, dim=0)
    encoder_attention_mask = (
        attention_mask.repeat_interleave(beams, dim=0) if attention_mask is not None else None
    )

    # Only beam 0 is alive at step 0, otherwise all beams would expand the same
    # start token and return num_beams identical hypotheses.
    beam_scores = torch.full((batch, beams), NEG_INF, dtype=torch.float32, device=device)
    beam_scores[:, 0] = 0.0
    beam_scores = beam_scores.view(-1)

    tokens = torch.zeros((rows, 0), dtype=torch.long, device=device)
    token_logprobs = torch.zeros((rows, 0), dtype=torch.float32, device=device)
    step_input_ids = torch.full(
        (rows, 1), config.decoder_start_token_id, dtype=torch.long, device=device
    )
    past_key_values: tuple | None = None

    hypotheses = [
        _BeamHypotheses(beams, config.length_penalty, config.early_stopping) for _ in range(batch)
    ]
    done = [False] * batch

    for step in range(config.max_length):
        logits, past_key_values = model.decode_step(
            step_input_ids,
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
        )
        step_logprobs = torch.log_softmax(logits[:, -1, :].float(), dim=-1)  # (rows, vocab)
        vocab = step_logprobs.shape[-1]

        candidate_scores = step_logprobs + beam_scores[:, None]
        candidate_scores = candidate_scores.view(batch, beams * vocab)
        top_scores, top_indices = candidate_scores.topk(min(2 * beams, beams * vocab), dim=-1)
        source_beam = torch.div(top_indices, vocab, rounding_mode="floor")
        candidate_tokens = top_indices % vocab

        next_scores = torch.full((batch, beams), NEG_INF, dtype=torch.float32, device=device)
        next_tokens = torch.full(
            (batch, beams), config.pad_token_id, dtype=torch.long, device=device
        )
        next_rows = torch.zeros((batch, beams), dtype=torch.long, device=device)

        for b in range(batch):
            if done[b]:
                # Frozen: keep the rows addressable so the cache stays aligned.
                next_scores[b] = NEG_INF
                next_rows[b] = torch.arange(b * beams, (b + 1) * beams, device=device)
                continue
            filled = 0
            for rank in range(top_scores.shape[1]):
                row = b * beams + int(source_beam[b, rank])
                token = int(candidate_tokens[b, rank])
                score = float(top_scores[b, rank])
                if token == config.eos_token_id:
                    hypotheses[b].add(
                        torch.cat([tokens[row], candidate_tokens[b, rank, None]]),
                        torch.cat([token_logprobs[row], step_logprobs[row, token, None]]),
                        score,
                    )
                else:
                    next_scores[b, filled] = score
                    next_tokens[b, filled] = token
                    next_rows[b, filled] = row
                    filled += 1
                if filled == beams:
                    break
            if filled < beams:
                # Fewer than num_beams non-EOS candidates: pad the beam with
                # dead rows (score -inf) so shapes stay static.
                next_rows[b, filled:] = b * beams
            done[b] = hypotheses[b].is_done(float(top_scores[b, 0]), step + 1)

        flat_rows = next_rows.view(-1)
        flat_tokens = next_tokens.view(-1)
        chosen_logprobs = step_logprobs[flat_rows, flat_tokens]
        tokens = torch.cat([tokens[flat_rows], flat_tokens[:, None]], dim=1)
        token_logprobs = torch.cat([token_logprobs[flat_rows], chosen_logprobs[:, None]], dim=1)
        beam_scores = next_scores.view(-1)
        assert past_key_values is not None
        past_key_values = _reorder_cache(past_key_values, flat_rows)
        step_input_ids = flat_tokens[:, None]

        if all(done):
            break

    # Finalise rows that never filled their hypothesis pool.
    for b in range(batch):
        if done[b]:
            continue
        for beam in range(beams):
            row = b * beams + beam
            score = float(beam_scores[row])
            if score == NEG_INF:
                continue
            hypotheses[b].add(tokens[row], token_logprobs[row], score)

    best = [max(h.beams, key=lambda item: item[0]) for h in hypotheses]
    lengths = torch.tensor([int(item[1].numel()) for item in best], device=device)
    width = int(lengths.max())

    sequences = torch.full((batch, width), config.pad_token_id, dtype=torch.long, device=device)
    out_logprobs = torch.zeros((batch, width), dtype=torch.float32, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    for b, (_, hyp_tokens, hyp_logprobs) in enumerate(best):
        n = int(hyp_tokens.numel())
        sequences[b, :n] = hyp_tokens
        out_logprobs[b, :n] = hyp_logprobs
        finished[b] = bool(n > 0 and int(hyp_tokens[-1]) == config.eos_token_id)

    return GenerationOutput(
        sequences=sequences,
        scores=torch.tensor([item[0] for item in best], dtype=torch.float32, device=device),
        token_logprobs=out_logprobs,
        finished=finished,
        lengths=lengths,
    )
