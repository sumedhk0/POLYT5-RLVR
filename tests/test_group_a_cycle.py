# tests/test_group_a_cycle.py
"""Group A Task 10: cycle consistency is a flag, and the flag is off."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.model.multitask import MultiTaskConfig, PolyT5MultiTask
from polyt5.tokenization import PolyT5Tokenizer
from polyt5.training.cycle import CycleConfig, build_cycle_loss
from polyt5.utils import seed_everything

REPO = Path(__file__).resolve().parents[1]
TINY_YAML = REPO / "configs" / "model" / "polyt5_tiny.yaml"
VOCAB = REPO / "artifacts" / "tokenizer" / "polyt5_vocab.json"

pytestmark = pytest.mark.skipif(not VOCAB.is_file(), reason="tokenizer artifact missing")


def build_model(*, regression_head: bool = True) -> tuple[PolyT5MultiTask, PolyT5Tokenizer]:
    seed_everything(0)
    tokenizer = PolyT5Tokenizer.from_file(VOCAB)
    config = PolyT5Config.from_yaml(TINY_YAML)
    config.dropout_rate = 0.0
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_id
    config.eos_token_id = tokenizer.eos_id
    config.decoder_start_token_id = tokenizer.decoder_start_token_id
    model = PolyT5MultiTask(
        PolyT5ForConditionalGeneration(config),
        MultiTaskConfig(use_regression_head=regression_head, head_dropout=0.0),
    )
    model.set_target_scaling(mean=417.0, std=113.0)
    return model, tokenizer


def test_the_default_is_off():
    assert CycleConfig().enabled is False


def test_disabled_config_builds_no_loss_at_all():
    model, tokenizer = build_model()
    assert build_cycle_loss(model, tokenizer, config=CycleConfig(), device="cpu") is None


def test_enabled_config_builds_a_callable_returning_a_scalar():
    model, tokenizer = build_model()
    loss_fn = build_cycle_loss(
        model, tokenizer, config=CycleConfig(enabled=True, max_length=16), device="cpu"
    )
    assert loss_fn is not None
    loss = loss_fn(torch.zeros(2))
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_the_cycle_loss_carries_a_gradient_into_the_encoder():
    model, tokenizer = build_model()
    loss_fn = build_cycle_loss(
        model, tokenizer, config=CycleConfig(enabled=True, max_length=16), device="cpu"
    )
    loss_fn(torch.zeros(2)).backward()
    grads = [p.grad for p in model.backbone.encoder.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0.0 for g in grads)


def test_a_model_without_a_regression_head_cannot_close_the_loop():
    model, tokenizer = build_model(regression_head=False)
    with pytest.raises(ValueError, match="regression head"):
        build_cycle_loss(model, tokenizer, config=CycleConfig(enabled=True), device="cpu")


def test_the_same_seed_reproduces_the_same_cycle_loss():
    model, tokenizer = build_model()
    config = CycleConfig(enabled=True, max_length=16, seed=5)
    first = build_cycle_loss(model, tokenizer, config=config, device="cpu")(torch.zeros(2))
    second = build_cycle_loss(model, tokenizer, config=config, device="cpu")(torch.zeros(2))
    assert float(first) == pytest.approx(float(second))


def test_an_empty_target_batch_returns_zero_rather_than_nan():
    model, tokenizer = build_model()
    loss_fn = build_cycle_loss(
        model, tokenizer, config=CycleConfig(enabled=True, max_length=16), device="cpu"
    )
    loss = loss_fn(torch.zeros(0))
    assert float(loss) == pytest.approx(0.0)


def test_degenerate_sampling_settings_are_refused():
    model, tokenizer = build_model()
    with pytest.raises(ValueError, match="temperature"):
        build_cycle_loss(
            model, tokenizer, config=CycleConfig(enabled=True, temperature=0.0), device="cpu"
        )
    with pytest.raises(ValueError, match="top_p"):
        build_cycle_loss(
            model, tokenizer, config=CycleConfig(enabled=True, top_p=0.0), device="cpu"
        )
