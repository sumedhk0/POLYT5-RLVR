# tests/test_group_a_multitask_model.py
"""Group A Task 6: the shared-encoder wrapper and its three forward paths."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.model.multitask import HeadOutput, MultiTaskConfig, PolyT5MultiTask
from polyt5.utils import seed_everything

REPO = Path(__file__).resolve().parents[1]
TINY_YAML = REPO / "configs" / "model" / "polyt5_tiny.yaml"


def tiny_backbone(seed: int = 0) -> PolyT5ForConditionalGeneration:
    seed_everything(seed)
    config = PolyT5Config.from_yaml(TINY_YAML)
    config.dropout_rate = 0.0
    return PolyT5ForConditionalGeneration(config)


def wrap(**kwargs) -> PolyT5MultiTask:
    seed_everything(0)
    return PolyT5MultiTask(
        tiny_backbone(), MultiTaskConfig(head_dropout=0.0, **kwargs)
    )


def batch(n: int = 3, length: int = 6, vocab: int = 458):
    generator = torch.Generator().manual_seed(11)
    return (
        torch.randint(2, vocab, (n, length), generator=generator),
        torch.ones(n, length, dtype=torch.long),
    )


def test_wrapper_exposes_the_backbone_config_so_checkpoints_stay_loadable():
    model = wrap(use_regression_head=True)
    assert isinstance(model.config, PolyT5Config)
    assert model.config is model.backbone.config
    assert PolyT5Config.from_dict(model.config.to_dict()) == model.config


def test_state_dict_is_namespaced_under_backbone():
    model = wrap(use_regression_head=True, n_descriptors=4)
    keys = list(model.state_dict())
    assert any(key.startswith("backbone.") for key in keys)
    assert any(key.startswith("tg_head.") for key in keys)
    assert any(key.startswith("descriptor_head.") for key in keys)
    assert "tg_mean" in keys and "tg_std" in keys


def test_generation_is_untouched():
    """Spec 4.1: 'Generation is untouched.' Bit-identical, not merely similar."""
    backbone = tiny_backbone()
    model = PolyT5MultiTask(backbone, MultiTaskConfig(use_regression_head=True,
                                                      head_dropout=0.0))
    model.eval()
    backbone.eval()
    input_ids, mask = batch()
    labels = torch.randint(2, 458, (3, 5), generator=torch.Generator().manual_seed(12))
    wrapped = model.forward_generation(input_ids, mask, labels)
    bare = backbone(input_ids=input_ids, attention_mask=mask, labels=labels)
    assert torch.equal(wrapped.logits, bare.logits)
    assert float(wrapped.loss.detach()) == pytest.approx(float(bare.loss.detach()))


def test_forward_text_without_extras_returns_the_backbone_loss_verbatim():
    """B0 must REPRODUCE the baseline, not approximate it."""
    backbone = tiny_backbone()
    model = PolyT5MultiTask(backbone, MultiTaskConfig(head_dropout=0.0))
    model.eval()
    backbone.eval()
    input_ids, mask = batch()
    labels = torch.tensor([[5, 6, -100], [7, 8, 9], [10, -100, -100]])
    out = model.forward_text(input_ids, mask, labels)
    bare = backbone(input_ids=input_ids, attention_mask=mask, labels=labels)
    assert isinstance(out, HeadOutput)
    assert float(out.loss.detach()) == pytest.approx(float(bare.loss.detach()), abs=1e-7)
    assert out.descriptor_loss is None


def test_regression_forward_produces_one_scalar_per_example():
    model = wrap(use_regression_head=True)
    model.eval()
    input_ids, mask = batch()
    out = model.forward_regression(input_ids, mask)
    assert out.tg_standardised.shape == (3,)
    assert out.tg_kelvin.shape == (3,)
    assert out.loss is None


def test_target_scaling_inverts_at_inference():
    model = wrap(use_regression_head=True)
    model.set_target_scaling(mean=417.0, std=113.0)
    model.eval()
    input_ids, mask = batch()
    out = model.forward_regression(input_ids, mask)
    assert torch.allclose(out.tg_kelvin, out.tg_standardised * 113.0 + 417.0, atol=1e-5)
    assert torch.allclose(model.predict_tg(input_ids, mask), out.tg_kelvin, atol=1e-5)


def test_set_target_scaling_refuses_a_degenerate_std():
    model = wrap(use_regression_head=True)
    with pytest.raises(ValueError, match="std"):
        model.set_target_scaling(mean=417.0, std=0.0)


def test_descriptor_loss_enters_with_lambda():
    model = wrap(use_regression_head=True, n_descriptors=4, descriptor_lambda=0.25)
    model.eval()
    input_ids, mask = batch()
    targets = torch.zeros(3)
    descriptors = torch.zeros(3, 4)
    out = model.forward_regression(
        input_ids, mask, tg_targets=targets, descriptor_targets=descriptors
    )
    assert out.tg_loss is not None and out.descriptor_loss is not None
    assert float(out.loss) == pytest.approx(
        float(out.tg_loss) + 0.25 * float(out.descriptor_loss), abs=1e-6
    )


def test_descriptor_auxiliaries_ride_on_the_text_path_too():
    """Arm A2 is descriptors WITHOUT a regression head; it must still work."""
    model = wrap(n_descriptors=4, descriptor_lambda=0.5)
    model.eval()
    input_ids, mask = batch()
    labels = torch.randint(2, 458, (3, 4), generator=torch.Generator().manual_seed(13))
    out = model.forward_text(input_ids, mask, labels, descriptor_targets=torch.zeros(3, 4))
    assert out.descriptor_loss is not None
    assert out.tg_loss is not None
    assert float(out.loss) == pytest.approx(
        float(out.tg_loss) + 0.5 * float(out.descriptor_loss), abs=1e-6
    )


def test_regression_forward_without_a_head_is_a_loud_error():
    model = wrap()
    input_ids, mask = batch()
    with pytest.raises(RuntimeError, match="use_regression_head"):
        model.forward_regression(input_ids, mask)


def test_gradients_reach_the_shared_encoder_from_the_regression_head():
    model = wrap(use_regression_head=True)
    input_ids, mask = batch()
    out = model.forward_regression(input_ids, mask, tg_targets=torch.zeros(3))
    out.loss.backward()
    encoder_grads = [
        p.grad for p in model.backbone.encoder.parameters() if p.grad is not None
    ]
    assert encoder_grads, "the point of the shared encoder is that it receives this gradient"
    assert any(float(g.abs().sum()) > 0.0 for g in encoder_grads)


def test_multitask_config_round_trips_through_dict():
    config = MultiTaskConfig(use_regression_head=True, n_descriptors=97,
                             descriptor_lambda=0.3, huber_delta=2.0, head_dropout=0.05)
    assert MultiTaskConfig.from_dict(config.to_dict()) == config
    assert MultiTaskConfig.from_dict({"n_descriptors": 5}).n_descriptors == 5


def test_num_parameters_counts_the_heads():
    plain = wrap()
    with_heads = wrap(use_regression_head=True, n_descriptors=8)
    assert with_heads.num_parameters() > plain.num_parameters()
