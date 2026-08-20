"""Tests for the from-scratch PolyT5 (T5) encoder-decoder implementation.

All tests are CPU-only and use tiny tensors so the whole file runs in seconds.
Ground truth for parameter counts is Table S2 of Sahu et al. (npj Artificial
Intelligence 2026): small 1.44M, medium 7.46M, large 58.98M.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pytest  # noqa: E402
import torch  # noqa: E402

from polyt5.model import (  # noqa: E402
    PolyT5Config,
    PolyT5ForConditionalGeneration,
    RelativePositionBias,
    relative_position_bucket,
)

CONFIG_DIR = REPO_ROOT / "configs" / "model"

# Paper-reported totals (Table S2), in raw parameters.
PAPER_PARAM_COUNTS = {
    "polyt5_small.yaml": 1.44e6,
    "polyt5_medium.yaml": 7.46e6,
    "polyt5_large.yaml": 58.98e6,
}


@pytest.fixture()
def tiny_config() -> PolyT5Config:
    return PolyT5Config.from_yaml(CONFIG_DIR / "polyt5_tiny.yaml")


@pytest.fixture()
def tiny_model(tiny_config: PolyT5Config) -> PolyT5ForConditionalGeneration:
    torch.manual_seed(0)
    model = PolyT5ForConditionalGeneration(tiny_config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# relative_position_bucket
# ---------------------------------------------------------------------------


class TestRelativePositionBucket:
    def test_bidirectional_hand_computed(self) -> None:
        # num_buckets=8 -> 4 per direction; max_exact=2; log range up to max_distance=16.
        rp = torch.tensor([-100, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 100])
        expected = torch.tensor([3, 3, 3, 2, 2, 1, 0, 5, 6, 6, 7, 7, 7])
        got = relative_position_bucket(rp, bidirectional=True, num_buckets=8, max_distance=16)
        assert torch.equal(got, expected)

    def test_causal_hand_computed(self) -> None:
        # Causal: only negative relative positions (past keys) are distinguished.
        rp = torch.tensor([-100, -16, -8, -4, -2, -1, 0, 1, 5, 100])
        expected = torch.tensor([7, 7, 6, 4, 2, 1, 0, 0, 0, 0])
        got = relative_position_bucket(rp, bidirectional=False, num_buckets=8, max_distance=16)
        assert torch.equal(got, expected)

    def test_causal_zero_maps_to_bucket_zero(self) -> None:
        rp = torch.tensor([0])
        got = relative_position_bucket(rp, bidirectional=False, num_buckets=32, max_distance=128)
        assert got.item() == 0

    def test_outputs_in_range(self) -> None:
        rp = torch.arange(-300, 300)
        for bidirectional in (True, False):
            got = relative_position_bucket(
                rp, bidirectional=bidirectional, num_buckets=32, max_distance=128
            )
            assert got.min().item() >= 0
            assert got.max().item() < 32

    def test_clamp_beyond_max_distance(self) -> None:
        # Distances at/ beyond max_distance saturate into the final bucket.
        rp = torch.tensor([-128, -500, -10_000])
        got = relative_position_bucket(rp, bidirectional=False, num_buckets=32, max_distance=128)
        assert torch.equal(got, torch.full((3,), 31, dtype=got.dtype))
        rp_bi = torch.tensor([128, 500, 10_000])
        got_bi = relative_position_bucket(
            rp_bi, bidirectional=True, num_buckets=32, max_distance=128
        )
        assert torch.equal(got_bi, torch.full((3,), 31, dtype=got_bi.dtype))

    def test_monotonic_in_distance_causal(self) -> None:
        # Buckets are non-decreasing as the (past) distance grows.
        rp = -torch.arange(0, 400)
        got = relative_position_bucket(rp, bidirectional=False, num_buckets=32, max_distance=128)
        diffs = got[1:] - got[:-1]
        assert (diffs >= 0).all()


# ---------------------------------------------------------------------------
# RelativePositionBias
# ---------------------------------------------------------------------------


class TestRelativePositionBias:
    def test_output_shape(self) -> None:
        bias = RelativePositionBias(
            num_buckets=8, max_distance=16, num_heads=2, bidirectional=True
        )
        out = bias(5, 7, device=torch.device("cpu"))
        assert out.shape == (1, 2, 5, 7)

    def test_broadcasts_against_scores(self) -> None:
        bias = RelativePositionBias(
            num_buckets=8, max_distance=16, num_heads=2, bidirectional=False
        )
        out = bias(4, 4, device=torch.device("cpu"))
        scores = torch.zeros(3, 2, 4, 4)
        combined = scores + out
        assert combined.shape == (3, 2, 4, 4)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_yaml_round_trip(self, tiny_config: PolyT5Config, tmp_path: Path) -> None:
        out = tmp_path / "roundtrip.yaml"
        tiny_config.save_yaml(out)
        reloaded = PolyT5Config.from_yaml(out)
        assert reloaded == tiny_config

    def test_dict_round_trip(self, tiny_config: PolyT5Config) -> None:
        assert PolyT5Config.from_dict(tiny_config.to_dict()) == tiny_config

    def test_num_decoder_layers_defaults_to_num_layers(self) -> None:
        cfg = PolyT5Config(num_layers=3, num_decoder_layers=None)
        assert cfg.num_decoder_layers_resolved == 3
        cfg2 = PolyT5Config(num_layers=3, num_decoder_layers=5)
        assert cfg2.num_decoder_layers_resolved == 5

    def test_invalid_feed_forward_proj_raises(self) -> None:
        with pytest.raises(ValueError):
            PolyT5Config(feed_forward_proj="swish")

    def test_inner_dim_mismatch_warns(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            PolyT5Config(d_model=100, d_kv=32, num_heads=4)  # 128 != 100
        assert any("d_kv" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# Forward passes
# ---------------------------------------------------------------------------


class TestForward:
    def test_encoder_shapes(
        self, tiny_model: PolyT5ForConditionalGeneration, tiny_config: PolyT5Config
    ) -> None:
        ids = torch.randint(2, tiny_config.vocab_size, (2, 7))
        enc = tiny_model.encode(ids, attention_mask=torch.ones_like(ids))
        assert enc.shape == (2, 7, tiny_config.d_model)
        assert enc.dtype == torch.float32

    def test_full_forward_logits_shape(
        self, tiny_model: PolyT5ForConditionalGeneration, tiny_config: PolyT5Config
    ) -> None:
        src = torch.randint(2, tiny_config.vocab_size, (2, 7))
        tgt = torch.randint(2, tiny_config.vocab_size, (2, 5))
        out = tiny_model(input_ids=src, decoder_input_ids=tgt)
        assert out.logits.shape == (2, 5, tiny_config.vocab_size)
        assert out.loss is None
        assert out.encoder_last_hidden_state is not None

    def test_cross_attention_consumes_encoder_states(
        self, tiny_model: PolyT5ForConditionalGeneration, tiny_config: PolyT5Config
    ) -> None:
        torch.manual_seed(1)
        tgt = torch.randint(2, tiny_config.vocab_size, (1, 4))
        enc_a = torch.randn(1, 6, tiny_config.d_model)
        enc_b = torch.randn(1, 6, tiny_config.d_model)
        out_a = tiny_model(input_ids=None, encoder_outputs=enc_a, decoder_input_ids=tgt)
        out_b = tiny_model(input_ids=None, encoder_outputs=enc_b, decoder_input_ids=tgt)
        assert not torch.allclose(out_a.logits, out_b.logits)


# ---------------------------------------------------------------------------
# Loss / labels
# ---------------------------------------------------------------------------


class TestLabels:
    def test_loss_scalar_and_backward(
        self, tiny_config: PolyT5Config
    ) -> None:
        torch.manual_seed(0)
        model = PolyT5ForConditionalGeneration(tiny_config)  # train mode for grads
        src = torch.randint(2, tiny_config.vocab_size, (2, 6))
        labels = torch.randint(2, tiny_config.vocab_size, (2, 4))
        out = model(input_ids=src, labels=labels)
        assert out.loss is not None
        assert out.loss.dim() == 0
        assert torch.isfinite(out.loss)
        assert out.loss.requires_grad
        out.loss.backward()
        assert model.shared.weight.grad is not None
        assert model.shared.weight.grad.abs().sum().item() > 0.0

    def test_ignore_index_changes_loss(
        self, tiny_model: PolyT5ForConditionalGeneration, tiny_config: PolyT5Config
    ) -> None:
        torch.manual_seed(2)
        src = torch.randint(2, tiny_config.vocab_size, (2, 6))
        labels = torch.randint(2, tiny_config.vocab_size, (2, 5))
        loss_full = tiny_model(input_ids=src, labels=labels).loss
        masked = labels.clone()
        masked[:, -2:] = -100
        loss_masked = tiny_model(input_ids=src, labels=masked).loss
        assert not torch.isclose(loss_full, loss_masked)

    def test_fully_masked_row_no_nan(
        self, tiny_model: PolyT5ForConditionalGeneration, tiny_config: PolyT5Config
    ) -> None:
        torch.manual_seed(3)
        src = torch.randint(2, tiny_config.vocab_size, (2, 6))
        labels = torch.randint(2, tiny_config.vocab_size, (2, 5))
        labels[0, :] = -100  # one row entirely ignored
        loss = tiny_model(input_ids=src, labels=labels).loss
        assert torch.isfinite(loss)

    def test_shift_right_contract(
        self, tiny_model: PolyT5ForConditionalGeneration, tiny_config: PolyT5Config
    ) -> None:
        labels = torch.tensor([[5, 6, 7, 1], [8, 9, -100, -100]])
        shifted = tiny_model._shift_right(labels)
        start = tiny_config.decoder_start_token_id
        pad = tiny_config.pad_token_id
        assert (shifted[:, 0] == start).all()
        expected_rest = labels[:, :-1].clone()
        expected_rest[expected_rest == -100] = pad
        assert torch.equal(shifted[:, 1:], expected_rest)
        assert (shifted != -100).all()


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


class TestMasks:
    def test_encoder_padding_does_not_change_logits(
        self, tiny_model: PolyT5ForConditionalGeneration, tiny_config: PolyT5Config
    ) -> None:
        torch.manual_seed(4)
        pad = tiny_config.pad_token_id
        src = torch.randint(2, tiny_config.vocab_size, (1, 5))
        tgt = torch.randint(2, tiny_config.vocab_size, (1, 4))
        out_short = tiny_model(
            input_ids=src, attention_mask=torch.ones_like(src), decoder_input_ids=tgt
        )
        src_padded = torch.cat([src, torch.full((1, 4), pad)], dim=1)
        mask = torch.cat([torch.ones_like(src), torch.zeros((1, 4), dtype=torch.long)], dim=1)
        out_padded = tiny_model(input_ids=src_padded, attention_mask=mask, decoder_input_ids=tgt)
        assert torch.allclose(out_short.logits, out_padded.logits, atol=1e-5)

    def test_causal_mask_blocks_future(
        self, tiny_model: PolyT5ForConditionalGeneration, tiny_config: PolyT5Config
    ) -> None:
        torch.manual_seed(5)
        src = torch.randint(2, tiny_config.vocab_size, (1, 5))
        tgt = torch.randint(2, tiny_config.vocab_size, (1, 6))
        out_a = tiny_model(input_ids=src, decoder_input_ids=tgt)
        tgt_b = tgt.clone()
        tgt_b[0, 4] = (tgt_b[0, 4] + 1) % tiny_config.vocab_size
        out_b = tiny_model(input_ids=src, decoder_input_ids=tgt_b)
        # Positions strictly before the edited token are unaffected.
        assert torch.allclose(out_a.logits[:, :4], out_b.logits[:, :4], atol=1e-6)
        # The edited position itself (and later) must be affected.
        assert not torch.allclose(out_a.logits[:, 4:], out_b.logits[:, 4:])


# ---------------------------------------------------------------------------
# Incremental decoding (cache path for future RL sampling)
# ---------------------------------------------------------------------------


class TestIncrementalDecoding:
    def test_decode_step_matches_full_forward(
        self, tiny_model: PolyT5ForConditionalGeneration, tiny_config: PolyT5Config
    ) -> None:
        torch.manual_seed(6)
        src = torch.randint(2, tiny_config.vocab_size, (1, 5))
        src_mask = torch.ones_like(src)
        tgt = torch.randint(2, tiny_config.vocab_size, (1, 4))
        full = tiny_model(input_ids=src, attention_mask=src_mask, decoder_input_ids=tgt)
        enc = tiny_model.encode(src, attention_mask=src_mask)
        past = None
        step_logits = []
        for t in range(tgt.shape[1]):
            logits, past = tiny_model.decode_step(
                decoder_input_ids=tgt[:, t : t + 1],
                encoder_hidden_states=enc,
                encoder_attention_mask=src_mask,
                past_key_values=past,
            )
            step_logits.append(logits)
        stepped = torch.cat(step_logits, dim=1)
        assert torch.allclose(full.logits, stepped, atol=1e-5)


# ---------------------------------------------------------------------------
# Parameter counts vs paper Table S2
# ---------------------------------------------------------------------------


class TestParameterCounts:
    @pytest.mark.parametrize("yaml_name,paper_count", sorted(PAPER_PARAM_COUNTS.items()))
    def test_within_10_percent_of_paper(self, yaml_name: str, paper_count: float) -> None:
        cfg = PolyT5Config.from_yaml(CONFIG_DIR / yaml_name)
        model = PolyT5ForConditionalGeneration(cfg)
        n = model.num_parameters()
        rel_err = abs(n - paper_count) / paper_count
        assert rel_err < 0.10, f"{yaml_name}: measured {n} vs paper {paper_count:.0f}"

    @pytest.mark.parametrize("yaml_name", sorted(PAPER_PARAM_COUNTS))
    def test_analytic_estimate_matches_actual(self, yaml_name: str) -> None:
        cfg = PolyT5Config.from_yaml(CONFIG_DIR / yaml_name)
        model = PolyT5ForConditionalGeneration(cfg)
        assert cfg.estimate_num_parameters() == model.num_parameters()

    def test_table_s2_values_in_yamls(self) -> None:
        expected = {
            "polyt5_small.yaml": dict(d_model=128, num_layers=3, d_ff=512, num_heads=4, d_kv=32),
            "polyt5_medium.yaml": dict(d_model=256, num_layers=4, d_ff=1024, num_heads=4, d_kv=64),
            "polyt5_large.yaml": dict(d_model=512, num_layers=8, d_ff=2048, num_heads=8, d_kv=64),
        }
        for name, fields in expected.items():
            cfg = PolyT5Config.from_yaml(CONFIG_DIR / name)
            for key, value in fields.items():
                assert getattr(cfg, key) == value, f"{name}: {key}"
            assert cfg.n_positions == 200


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_logits(self, tiny_config: PolyT5Config) -> None:
        src = torch.randint(2, tiny_config.vocab_size, (2, 6))
        tgt = torch.randint(2, tiny_config.vocab_size, (2, 4))

        def build_and_run() -> torch.Tensor:
            torch.manual_seed(42)
            model = PolyT5ForConditionalGeneration(tiny_config)
            model.eval()
            with torch.no_grad():
                return model(input_ids=src, decoder_input_ids=tgt).logits

        assert torch.equal(build_and_run(), build_and_run())
