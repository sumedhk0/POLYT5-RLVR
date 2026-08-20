"""Tests for the decoding stack (sampling, greedy/sampled generation, beam search).

All tests are CPU-only and use the deliberately tiny debug model from
``configs/model/polyt5_tiny.yaml`` so the whole file runs in seconds.

Ground truth for the decoding regimes comes from Sahu et al. (npj Artificial
Intelligence 2026):

* Property prediction decodes with BEAM SEARCH, beam width 4.
* Conditional generation decodes with SAMPLING ("instead of beam search"),
  ``top_p`` in {0.75, 0.95} and temperature swept 0.1 -> 2.0, max output
  length 200 tokens.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pytest  # noqa: E402
import torch  # noqa: E402

from polyt5.generation import (  # noqa: E402
    BeamSearchConfig,
    GenerationConfig,
    apply_repetition_penalty,
    apply_temperature,
    batch_generate,
    beam_search,
    generate,
    length_penalized_score,
    top_k_filter,
    top_p_filter,
)
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs" / "model"
NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_config() -> PolyT5Config:
    return PolyT5Config.from_yaml(CONFIG_DIR / "polyt5_tiny.yaml")


@pytest.fixture()
def tiny_model(tiny_config: PolyT5Config) -> PolyT5ForConditionalGeneration:
    torch.manual_seed(0)
    model = PolyT5ForConditionalGeneration(tiny_config)
    model.eval()  # generate() never changes the module mode; dropout must be off
    return model


@pytest.fixture()
def source(tiny_config: PolyT5Config) -> tuple[torch.Tensor, torch.Tensor]:
    """A tiny 2-row encoder batch with one padded row."""
    input_ids = torch.tensor([[5, 6, 7, 8], [9, 10, 11, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    return input_ids, attention_mask


def make_gen_config(cfg: PolyT5Config, **overrides: object) -> GenerationConfig:
    """Build a GenerationConfig carrying the model's special-token ids."""
    kwargs: dict[str, object] = {
        "eos_token_id": cfg.eos_token_id,
        "pad_token_id": cfg.pad_token_id,
        "decoder_start_token_id": cfg.decoder_start_token_id,
    }
    kwargs.update(overrides)
    return GenerationConfig(**kwargs)  # type: ignore[arg-type]


def patch_logit_bias(
    monkeypatch: pytest.MonkeyPatch,
    model: PolyT5ForConditionalGeneration,
    bias: torch.Tensor,
) -> None:
    """Add ``bias`` to every LM-head output.

    The LM head is bias-free and tied to the embedding, so the only clean
    injection point is :meth:`PolyT5ForConditionalGeneration._project_to_vocab`,
    which both ``forward`` and ``decode_step`` funnel through.

    Args:
        monkeypatch: pytest fixture.
        model: Model to patch.
        bias: Broadcastable to ``(batch, seq, vocab)``.
    """
    original = model._project_to_vocab

    def patched(hidden_states: torch.Tensor) -> torch.Tensor:
        return original(hidden_states) + bias.to(hidden_states.dtype)

    monkeypatch.setattr(model, "_project_to_vocab", patched)


def logits_from_probs(probs: list[float]) -> torch.Tensor:
    """Return ``(1, vocab)`` logits whose softmax is exactly ``probs``."""
    return torch.log(torch.tensor([probs], dtype=torch.float64)).float()


def kept_indices(filtered: torch.Tensor) -> set[int]:
    """Indices of row 0 that survived filtering (were not set to -inf)."""
    return {i for i, v in enumerate(filtered[0].tolist()) if v != NEG_INF}


def entropy(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    return float(-(probs * torch.log(probs.clamp_min(1e-30))).sum())


def strip_to_eos(row: list[int], eos: int, pad: int) -> list[int]:
    """Truncate a generated row at (and including) the first EOS, dropping pads."""
    out: list[int] = []
    for token in row:
        out.append(token)
        if token == eos:
            return out
    return [t for t in out if t != pad] if pad in out else out


# ---------------------------------------------------------------------------
# sampling.py — pure logit processors
# ---------------------------------------------------------------------------


class TestTopPFilter:
    def test_keeps_smallest_set_reaching_top_p(self) -> None:
        logits = logits_from_probs([0.5, 0.3, 0.15, 0.05])
        # cumulative: 0.5, 0.8, 0.95, 1.0 -> the token that CROSSES 0.75 is kept.
        assert kept_indices(top_p_filter(logits, 0.75)) == {0, 1}

    def test_crossing_token_is_kept_not_dropped(self) -> None:
        logits = logits_from_probs([0.6, 0.3, 0.07, 0.03])
        # cumulative: 0.6, 0.9, ... -> 0.6 alone is below 0.75, so index 1 is
        # required to reach it and must survive (the classic off-by-one).
        assert kept_indices(top_p_filter(logits, 0.75)) == {0, 1}

    def test_single_token_already_exceeds_top_p(self) -> None:
        logits = logits_from_probs([0.9, 0.05, 0.03, 0.02])
        assert kept_indices(top_p_filter(logits, 0.75)) == {0}

    def test_min_tokens_to_keep_rescues_boundary_case(self) -> None:
        logits = logits_from_probs([0.9, 0.05, 0.03, 0.02])
        assert kept_indices(top_p_filter(logits, 0.75, min_tokens_to_keep=2)) == {0, 1}

    def test_top_p_one_keeps_everything(self) -> None:
        logits = torch.randn(3, 11)
        filtered = top_p_filter(logits, 1.0)
        assert torch.equal(filtered, logits)

    def test_tiny_top_p_keeps_exactly_one(self) -> None:
        logits = torch.randn(4, 20)
        filtered = top_p_filter(logits, 1e-6)
        assert (filtered != NEG_INF).sum(dim=-1).tolist() == [1, 1, 1, 1]
        assert torch.equal(filtered.argmax(dim=-1), logits.argmax(dim=-1))

    def test_is_row_independent(self) -> None:
        logits = torch.cat(
            [logits_from_probs([0.9, 0.05, 0.03, 0.02]), logits_from_probs([0.4, 0.3, 0.2, 0.1])]
        )
        filtered = top_p_filter(logits, 0.75)
        assert (filtered[0] != NEG_INF).sum().item() == 1
        assert (filtered[1] != NEG_INF).sum().item() == 3  # 0.4, 0.7, 0.9 crosses at index 2

    def test_does_not_mutate_input(self) -> None:
        logits = torch.randn(2, 9)
        before = logits.clone()
        top_p_filter(logits, 0.5)
        assert torch.equal(logits, before)

    def test_rejects_invalid_top_p(self) -> None:
        with pytest.raises(ValueError):
            top_p_filter(torch.randn(1, 4), 0.0)
        with pytest.raises(ValueError):
            top_p_filter(torch.randn(1, 4), 1.5)


class TestApplyTemperature:
    def test_identity_at_one(self) -> None:
        logits = torch.randn(2, 17)
        assert torch.equal(apply_temperature(logits, 1.0), logits)

    def test_low_temperature_sharpens(self) -> None:
        logits = torch.randn(1, 32)
        assert entropy(apply_temperature(logits, 0.5)) < entropy(logits)

    def test_high_temperature_flattens(self) -> None:
        logits = torch.randn(1, 32)
        assert entropy(apply_temperature(logits, 2.0)) > entropy(logits)

    def test_paper_grid_is_monotone_in_entropy(self) -> None:
        # The paper sweeps T = 0.1 .. 2.0 in steps of 0.1; entropy must rise
        # monotonically along that sweep.
        torch.manual_seed(0)
        logits = torch.randn(1, 64)
        temps = [round(0.1 * i, 1) for i in range(1, 21)]
        entropies = [entropy(apply_temperature(logits, t)) for t in temps]
        assert all(a < b for a, b in zip(entropies[:-1], entropies[1:], strict=True))

    def test_non_positive_temperature_signals_greedy(self) -> None:
        logits = torch.tensor([[1.0, 5.0, 2.0, -3.0]])
        out = apply_temperature(logits, 0.0)
        assert out[0, 1].item() == 0.0
        assert kept_indices(out) == {1}
        # sampling from it is greedy
        assert torch.softmax(out, dim=-1)[0, 1].item() == pytest.approx(1.0)

    def test_does_not_mutate_input(self) -> None:
        logits = torch.randn(2, 9)
        before = logits.clone()
        apply_temperature(logits, 0.3)
        assert torch.equal(logits, before)


class TestTopKFilter:
    def test_keeps_exactly_k(self) -> None:
        logits = torch.randn(3, 40)
        filtered = top_k_filter(logits, 5)
        assert (filtered != NEG_INF).sum(dim=-1).tolist() == [5, 5, 5]

    def test_keeps_the_largest_k(self) -> None:
        logits = torch.tensor([[0.0, 3.0, 1.0, 2.0, -1.0]])
        assert kept_indices(top_k_filter(logits, 3)) == {1, 2, 3}

    def test_k_larger_than_vocab_is_noop(self) -> None:
        logits = torch.randn(2, 6)
        assert torch.equal(top_k_filter(logits, 100), logits)

    def test_rejects_non_positive_k(self) -> None:
        with pytest.raises(ValueError):
            top_k_filter(torch.randn(1, 4), 0)

    def test_does_not_mutate_input(self) -> None:
        logits = torch.randn(2, 9)
        before = logits.clone()
        top_k_filter(logits, 2)
        assert torch.equal(logits, before)


class TestRepetitionPenalty:
    def test_penalizes_seen_tokens_both_signs(self) -> None:
        logits = torch.tensor([[2.0, -2.0, 0.5, 1.0]])
        generated = torch.tensor([[0, 1]])
        out = apply_repetition_penalty(logits, generated, 2.0)
        assert out[0, 0].item() == pytest.approx(1.0)  # positive -> divided
        assert out[0, 1].item() == pytest.approx(-4.0)  # negative -> multiplied
        assert out[0, 2].item() == pytest.approx(0.5)  # untouched
        assert out[0, 3].item() == pytest.approx(1.0)

    def test_penalty_one_is_identity(self) -> None:
        logits = torch.randn(2, 8)
        generated = torch.tensor([[1, 2], [3, 4]])
        assert torch.equal(apply_repetition_penalty(logits, generated, 1.0), logits)

    def test_does_not_mutate_input(self) -> None:
        logits = torch.randn(2, 9)
        before = logits.clone()
        apply_repetition_penalty(logits, torch.tensor([[0], [1]]), 1.5)
        assert torch.equal(logits, before)


# ---------------------------------------------------------------------------
# GenerationConfig validation
# ---------------------------------------------------------------------------


class TestGenerationConfig:
    def test_paper_defaults(self, tiny_config: PolyT5Config) -> None:
        cfg = make_gen_config(tiny_config)
        assert cfg.max_length == 200  # paper: "maximum output length of 200 tokens"
        assert cfg.top_p == 0.95  # paper sweeps {0.75, 0.95}
        assert cfg.do_sample is True

    @pytest.mark.parametrize(
        "bad",
        [
            {"top_p": 0.0},
            {"top_p": 1.5},
            {"max_length": 0},
            {"max_length": 3, "min_length": 5},
            {"temperature": -0.5},
            {"top_k": 0},
            {"repetition_penalty": 0.0},
            {"num_return_sequences": 0},
            {"min_length": -1},
        ],
    )
    def test_rejects_invalid(self, tiny_config: PolyT5Config, bad: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            make_gen_config(tiny_config, **bad)


# ---------------------------------------------------------------------------
# greedy decoding
# ---------------------------------------------------------------------------


class TestGreedy:
    def test_is_deterministic(
        self, tiny_model: PolyT5ForConditionalGeneration, source: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        input_ids, mask = source
        cfg = make_gen_config(tiny_model.config, do_sample=False, max_length=10)
        a = generate(tiny_model, input_ids, mask, config=cfg)
        b = generate(tiny_model, input_ids, mask, config=cfg)
        assert torch.equal(a.sequences, b.sequences)
        assert torch.allclose(a.token_logprobs, b.token_logprobs)

    def test_first_token_equals_argmax_of_model_logits(
        self, tiny_model: PolyT5ForConditionalGeneration, source: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        input_ids, mask = source
        start = torch.full((input_ids.shape[0], 1), tiny_model.config.decoder_start_token_id)
        with torch.no_grad():
            out = tiny_model(input_ids, attention_mask=mask, decoder_input_ids=start)
        expected = out.logits[:, -1, :].argmax(dim=-1)

        cfg = make_gen_config(tiny_model.config, do_sample=False, max_length=4)
        got = generate(tiny_model, input_ids, mask, config=cfg)
        assert torch.equal(got.sequences[:, 0], expected)

    def test_kv_cache_equivalence(
        self, tiny_model: PolyT5ForConditionalGeneration, source: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        input_ids, mask = source
        cached = generate(
            tiny_model,
            input_ids,
            mask,
            config=make_gen_config(
                tiny_model.config, do_sample=False, max_length=12, use_cache=True
            ),
        )
        uncached = generate(
            tiny_model,
            input_ids,
            mask,
            config=make_gen_config(
                tiny_model.config, do_sample=False, max_length=12, use_cache=False
            ),
        )
        assert torch.equal(cached.sequences, uncached.sequences)
        assert torch.equal(cached.lengths, uncached.lengths)
        assert torch.allclose(cached.token_logprobs, uncached.token_logprobs, atol=1e-5)

    def test_logprobs_match_model_distribution_at_first_step(
        self, tiny_model: PolyT5ForConditionalGeneration, source: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        # token_logprobs come from the UNMODIFIED distribution (pre temperature /
        # top-p), which is what a policy-gradient method needs.
        input_ids, mask = source
        start = torch.full((input_ids.shape[0], 1), tiny_model.config.decoder_start_token_id)
        with torch.no_grad():
            out = tiny_model(input_ids, attention_mask=mask, decoder_input_ids=start)
        reference = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)

        cfg = make_gen_config(tiny_model.config, do_sample=False, max_length=3, temperature=0.4)
        got = generate(tiny_model, input_ids, mask, config=cfg)
        expected = reference.gather(1, got.sequences[:, :1]).squeeze(1)
        assert torch.allclose(got.token_logprobs[:, 0], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# EOS / length handling
# ---------------------------------------------------------------------------


class TestStopping:
    def test_always_eos_model_stops_immediately(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_model: PolyT5ForConditionalGeneration,
        source: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        cfg = tiny_model.config
        bias = torch.zeros(cfg.vocab_size)
        bias[cfg.eos_token_id] = 1e4
        patch_logit_bias(monkeypatch, tiny_model, bias)

        input_ids, mask = source
        out = generate(
            tiny_model,
            input_ids,
            mask,
            config=make_gen_config(cfg, do_sample=False, max_length=8),
        )
        assert out.sequences.shape == (2, 1)
        assert (out.sequences == cfg.eos_token_id).all()
        assert out.finished.all()
        assert out.lengths.tolist() == [1, 1]

    def test_min_length_suppresses_eos(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_model: PolyT5ForConditionalGeneration,
        source: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        cfg = tiny_model.config
        bias = torch.zeros(cfg.vocab_size)
        bias[cfg.eos_token_id] = 1e4
        patch_logit_bias(monkeypatch, tiny_model, bias)

        input_ids, mask = source
        out = generate(
            tiny_model,
            input_ids,
            mask,
            config=make_gen_config(cfg, do_sample=False, max_length=8, min_length=5),
        )
        assert out.sequences.shape == (2, 5)
        assert (out.sequences[:, :4] != cfg.eos_token_id).all()
        assert (out.sequences[:, 4] == cfg.eos_token_id).all()
        assert out.lengths.tolist() == [5, 5]

    def test_batch_rows_finish_independently(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_model: PolyT5ForConditionalGeneration,
        source: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        cfg = tiny_model.config
        # Row 0 is forced to emit EOS at once; row 1 must be untouched.
        bias = torch.zeros(2, 1, cfg.vocab_size)
        bias[0, 0, cfg.eos_token_id] = 1e4
        bias[1, 0, cfg.eos_token_id] = -1e4
        patch_logit_bias(monkeypatch, tiny_model, bias)

        input_ids, mask = source
        gen_cfg = make_gen_config(cfg, do_sample=False, max_length=7)
        out = generate(tiny_model, input_ids, mask, config=gen_cfg)

        assert out.sequences.shape == (2, 7)
        assert out.sequences[0, 0].item() == cfg.eos_token_id
        assert (out.sequences[0, 1:] == cfg.pad_token_id).all()  # frozen, not corrupted
        assert out.finished.tolist() == [True, False]
        assert out.lengths.tolist() == [1, 7]
        assert (out.sequences[1] != cfg.eos_token_id).all()
        assert (out.token_logprobs[0, 1:] == 0.0).all()  # pads contribute nothing
        assert out.scores[0].item() == pytest.approx(out.token_logprobs[0, 0].item())
        # The unfinished row keeps accumulating real log-probs after row 0 froze.
        assert (out.token_logprobs[1] < 0.0).all()

        # Row 1 must be bit-identical to a run where NO row finishes early
        # (same source, same per-row bias, different neighbour).
        control = generate(tiny_model, input_ids[[1, 1]], mask[[1, 1]], config=gen_cfg)
        assert torch.equal(control.sequences[1], out.sequences[1])
        assert torch.allclose(control.token_logprobs[1], out.token_logprobs[1])

    def test_never_exceeds_max_length(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_model: PolyT5ForConditionalGeneration,
        source: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        cfg = tiny_model.config
        bias = torch.zeros(cfg.vocab_size)
        bias[cfg.eos_token_id] = -1e4  # EOS never wins
        patch_logit_bias(monkeypatch, tiny_model, bias)

        input_ids, mask = source
        for max_length in (1, 3, 9):
            out = generate(
                tiny_model,
                input_ids,
                mask,
                config=make_gen_config(cfg, do_sample=False, max_length=max_length),
            )
            assert out.sequences.shape[1] == max_length
            assert not out.finished.any()
            assert out.lengths.tolist() == [max_length, max_length]


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


class TestSampling:
    def test_seed_is_deterministic_and_seeds_differ(
        self, tiny_model: PolyT5ForConditionalGeneration, source: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        input_ids, mask = source
        base = dict(do_sample=True, max_length=12, temperature=1.2, top_p=0.95)
        a = generate(
            tiny_model, input_ids, mask, config=make_gen_config(tiny_model.config, seed=7, **base)
        )
        b = generate(
            tiny_model, input_ids, mask, config=make_gen_config(tiny_model.config, seed=7, **base)
        )
        c = generate(
            tiny_model, input_ids, mask, config=make_gen_config(tiny_model.config, seed=8, **base)
        )
        assert torch.equal(a.sequences, b.sequences)
        assert not torch.equal(a.sequences, c.sequences)

    def test_explicit_generator_is_used_not_global_rng(
        self, tiny_model: PolyT5ForConditionalGeneration, source: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        input_ids, mask = source
        cfg = make_gen_config(tiny_model.config, do_sample=True, max_length=10, temperature=1.5)

        gen = torch.Generator().manual_seed(1234)
        torch.manual_seed(0)
        first = generate(tiny_model, input_ids, mask, config=cfg, generator=gen)

        gen = torch.Generator().manual_seed(1234)
        torch.manual_seed(999)  # global RNG perturbed: must not matter
        second = generate(tiny_model, input_ids, mask, config=cfg, generator=gen)
        assert torch.equal(first.sequences, second.sequences)

    def test_near_zero_temperature_matches_greedy(
        self, tiny_model: PolyT5ForConditionalGeneration, source: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        input_ids, mask = source
        greedy = generate(
            tiny_model,
            input_ids,
            mask,
            config=make_gen_config(tiny_model.config, do_sample=False, max_length=10),
        )
        sampled = generate(
            tiny_model,
            input_ids,
            mask,
            config=make_gen_config(
                tiny_model.config, do_sample=True, max_length=10, temperature=1e-4, seed=3
            ),
        )
        assert torch.equal(greedy.sequences, sampled.sequences)

    def test_num_return_sequences_expands_batch(
        self, tiny_model: PolyT5ForConditionalGeneration, source: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        input_ids, mask = source
        out = generate(
            tiny_model,
            input_ids,
            mask,
            config=make_gen_config(
                tiny_model.config,
                do_sample=True,
                max_length=10,
                temperature=1.5,
                top_p=0.95,
                num_return_sequences=3,
                seed=11,
            ),
        )
        assert out.sequences.shape[0] == 6  # 2 inputs x 3 samples, grouped per input
        rows = {tuple(r) for r in out.sequences.tolist()}
        assert len(rows) > 1  # sampling must not collapse to one sequence

    def test_token_logprobs_are_valid(
        self, tiny_model: PolyT5ForConditionalGeneration, source: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        input_ids, mask = source
        out = generate(
            tiny_model,
            input_ids,
            mask,
            config=make_gen_config(
                tiny_model.config, do_sample=True, max_length=9, temperature=1.0, seed=5
            ),
        )
        assert out.token_logprobs is not None
        assert out.token_logprobs.shape == out.sequences.shape
        assert torch.isfinite(out.token_logprobs).all()
        assert (out.token_logprobs <= 0.0).all()
        assert torch.allclose(out.scores, out.token_logprobs.sum(dim=-1), atol=1e-5)

    def test_top_k_restricts_support(
        self, tiny_model: PolyT5ForConditionalGeneration, source: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        input_ids, mask = source
        out = generate(
            tiny_model,
            input_ids,
            mask,
            config=make_gen_config(
                tiny_model.config, do_sample=True, max_length=6, top_k=1, top_p=1.0, seed=2
            ),
        )
        greedy = generate(
            tiny_model,
            input_ids,
            mask,
            config=make_gen_config(tiny_model.config, do_sample=False, max_length=6),
        )
        assert torch.equal(out.sequences, greedy.sequences)  # top_k=1 == greedy

    @pytest.mark.parametrize("top_p,temperature", [(0.75, 0.9), (0.75, 1.1), (0.95, 1.1)])
    def test_paper_grid_corners_smoke(
        self,
        tiny_model: PolyT5ForConditionalGeneration,
        source: tuple[torch.Tensor, torch.Tensor],
        top_p: float,
        temperature: float,
    ) -> None:
        """The paper's reported best (temperature, top_p) settings must run."""
        input_ids, mask = source
        out = generate(
            tiny_model,
            input_ids,
            mask,
            config=make_gen_config(
                tiny_model.config,
                do_sample=True,
                max_length=16,
                top_p=top_p,
                temperature=temperature,
                seed=0,
            ),
        )
        assert out.sequences.shape[0] == 2
        assert 1 <= out.sequences.shape[1] <= 16
        assert out.token_logprobs.shape == out.sequences.shape
        assert out.finished.dtype == torch.bool
        assert (out.lengths >= 1).all()


# ---------------------------------------------------------------------------
# batch_generate
# ---------------------------------------------------------------------------


class TestBatchGenerate:
    def test_chunks_and_strips_specials(
        self, tiny_model: PolyT5ForConditionalGeneration
    ) -> None:
        cfg = tiny_model.config
        prompts = [[5, 6, 7], [8, 9], [10, 11, 12, 13], [14], [15, 16]]
        out = batch_generate(
            tiny_model,
            prompts,
            config=make_gen_config(cfg, do_sample=False, max_length=6),
            pad_id=cfg.pad_token_id,
            device=torch.device("cpu"),
            batch_size=2,
        )
        assert len(out) == len(prompts)
        assert all(isinstance(row, list) for row in out)
        for row in out:
            assert cfg.eos_token_id not in row
            assert len(row) <= 6

    def test_chunking_does_not_change_results(
        self, tiny_model: PolyT5ForConditionalGeneration
    ) -> None:
        cfg = tiny_model.config
        prompts = [[5, 6, 7], [5, 6, 7], [5, 6, 7], [5, 6, 7]]
        gen_cfg = make_gen_config(cfg, do_sample=False, max_length=5)
        one = batch_generate(
            tiny_model,
            prompts,
            config=gen_cfg,
            pad_id=cfg.pad_token_id,
            device=torch.device("cpu"),
            batch_size=1,
        )
        four = batch_generate(
            tiny_model,
            prompts,
            config=gen_cfg,
            pad_id=cfg.pad_token_id,
            device=torch.device("cpu"),
            batch_size=4,
        )
        assert one == four


# ---------------------------------------------------------------------------
# beam search (paper: property prediction, beam width 4)
# ---------------------------------------------------------------------------


class TestBeamSearch:
    def test_length_penalty_direction(self) -> None:
        # Same total log-prob, different lengths: a larger length_penalty must
        # favour the LONGER sequence, a zero penalty the shorter one.
        short = (-6.0, 3)
        long = (-8.0, 8)
        assert length_penalized_score(*short, 0.0) > length_penalized_score(*long, 0.0)
        assert length_penalized_score(*long, 2.0) > length_penalized_score(*short, 2.0)

    def test_width_one_equals_greedy(
        self, tiny_model: PolyT5ForConditionalGeneration, source: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        cfg = tiny_model.config
        input_ids, mask = source
        greedy = generate(
            tiny_model,
            input_ids,
            mask,
            config=make_gen_config(cfg, do_sample=False, max_length=10),
        )
        beam = beam_search(
            tiny_model,
            input_ids,
            mask,
            config=BeamSearchConfig(
                num_beams=1,
                max_length=10,
                length_penalty=0.0,
                eos_token_id=cfg.eos_token_id,
                pad_token_id=cfg.pad_token_id,
                decoder_start_token_id=cfg.decoder_start_token_id,
            ),
        )
        for row in range(input_ids.shape[0]):
            g = strip_to_eos(greedy.sequences[row].tolist(), cfg.eos_token_id, cfg.pad_token_id)
            b = strip_to_eos(beam.sequences[row].tolist(), cfg.eos_token_id, cfg.pad_token_id)
            assert g == b

    def test_width_four_scores_at_least_greedy(
        self, tiny_model: PolyT5ForConditionalGeneration, source: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        cfg = tiny_model.config
        input_ids, mask = source
        greedy = generate(
            tiny_model,
            input_ids,
            mask,
            config=make_gen_config(cfg, do_sample=False, max_length=10),
        )
        beam = beam_search(
            tiny_model,
            input_ids,
            mask,
            config=BeamSearchConfig(
                num_beams=4,  # the paper's setting for property prediction
                max_length=10,
                length_penalty=0.0,  # raw summed log-prob, comparable with greedy
                eos_token_id=cfg.eos_token_id,
                pad_token_id=cfg.pad_token_id,
                decoder_start_token_id=cfg.decoder_start_token_id,
            ),
        )
        assert beam.sequences.shape[0] == input_ids.shape[0]
        assert (beam.scores + 1e-5 >= greedy.scores).all()

    def test_is_deterministic(
        self, tiny_model: PolyT5ForConditionalGeneration, source: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        cfg = tiny_model.config
        input_ids, mask = source
        bs_cfg = BeamSearchConfig(
            num_beams=4,
            max_length=8,
            eos_token_id=cfg.eos_token_id,
            pad_token_id=cfg.pad_token_id,
            decoder_start_token_id=cfg.decoder_start_token_id,
        )
        a = beam_search(tiny_model, input_ids, mask, config=bs_cfg)
        b = beam_search(tiny_model, input_ids, mask, config=bs_cfg)
        assert torch.equal(a.sequences, b.sequences)
        assert torch.allclose(a.scores, b.scores)

    def test_stops_at_eos_and_pads(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_model: PolyT5ForConditionalGeneration,
        source: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        cfg = tiny_model.config
        bias = torch.zeros(cfg.vocab_size)
        bias[cfg.eos_token_id] = 1e4
        patch_logit_bias(monkeypatch, tiny_model, bias)

        input_ids, mask = source
        out = beam_search(
            tiny_model,
            input_ids,
            mask,
            config=BeamSearchConfig(
                num_beams=4,
                max_length=9,
                eos_token_id=cfg.eos_token_id,
                pad_token_id=cfg.pad_token_id,
                decoder_start_token_id=cfg.decoder_start_token_id,
            ),
        )
        assert out.sequences.shape == (2, 1)
        assert (out.sequences[:, 0] == cfg.eos_token_id).all()
        assert out.finished.all()
        assert out.lengths.tolist() == [1, 1]

    def test_longer_length_penalty_does_not_shorten(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_model: PolyT5ForConditionalGeneration,
        source: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        cfg = tiny_model.config
        bias = torch.zeros(cfg.vocab_size)
        bias[cfg.eos_token_id] = 2.0  # EOS plausible but not forced
        patch_logit_bias(monkeypatch, tiny_model, bias)

        input_ids, mask = source

        def run(length_penalty: float) -> torch.Tensor:
            return beam_search(
                tiny_model,
                input_ids,
                mask,
                config=BeamSearchConfig(
                    num_beams=4,
                    max_length=10,
                    length_penalty=length_penalty,
                    eos_token_id=cfg.eos_token_id,
                    pad_token_id=cfg.pad_token_id,
                    decoder_start_token_id=cfg.decoder_start_token_id,
                ),
            ).lengths

        assert float(run(2.0).float().mean()) >= float(run(0.0).float().mean())

    def test_scores_are_finite_and_negative(
        self, tiny_model: PolyT5ForConditionalGeneration, source: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        cfg = tiny_model.config
        input_ids, mask = source
        out = beam_search(
            tiny_model,
            input_ids,
            mask,
            config=BeamSearchConfig(
                num_beams=4,
                max_length=8,
                eos_token_id=cfg.eos_token_id,
                pad_token_id=cfg.pad_token_id,
                decoder_start_token_id=cfg.decoder_start_token_id,
            ),
        )
        assert torch.isfinite(out.scores).all()
        assert (out.scores <= 0.0).all()
        assert out.token_logprobs.shape == out.sequences.shape
        assert math.isfinite(float(out.token_logprobs.sum()))

    def test_rejects_invalid_config(self, tiny_config: PolyT5Config) -> None:
        with pytest.raises(ValueError):
            BeamSearchConfig(
                num_beams=0,
                eos_token_id=tiny_config.eos_token_id,
                pad_token_id=tiny_config.pad_token_id,
                decoder_start_token_id=tiny_config.decoder_start_token_id,
            )
        with pytest.raises(ValueError):
            BeamSearchConfig(
                num_beams=4,
                max_length=0,
                eos_token_id=tiny_config.eos_token_id,
                pad_token_id=tiny_config.pad_token_id,
                decoder_start_token_id=tiny_config.decoder_start_token_id,
            )
