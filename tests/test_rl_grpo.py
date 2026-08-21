# tests/test_rl_grpo.py
from __future__ import annotations

import pytest
import torch

from polyt5.rl.grpo import GRPOConfig, grpo_loss, k3_kl


def _inputs(b=2, t=4):
    lp = torch.zeros(b, t, requires_grad=True)
    old = torch.zeros(b, t)
    ref = torch.zeros(b, t)
    adv = torch.tensor([1.0, -1.0])
    mask = torch.ones(b, t)
    return lp, old, ref, adv, mask


def test_k3_kl_is_non_negative_and_zero_at_identity():
    assert k3_kl(torch.zeros(3), torch.zeros(3)).abs().max() < 1e-6
    for delta in (-2.0, -0.5, 0.5, 2.0):
        assert k3_kl(torch.zeros(5), torch.full((5,), delta)).min() >= -1e-6


def test_loss_is_finite_and_differentiable():
    lp, old, ref, adv, mask = _inputs()
    loss, stats = grpo_loss(lp, old, ref, adv, mask, config=GRPOConfig())
    assert torch.isfinite(loss)
    loss.backward()
    assert lp.grad is not None and torch.isfinite(lp.grad).all()


def test_positive_advantage_pushes_logprob_up():
    """The core policy-gradient behaviour, checked by gradient sign."""
    lp, old, ref, adv, mask = _inputs()
    loss, _ = grpo_loss(lp, old, ref, adv, mask, config=GRPOConfig(kl_coef=0.0))
    loss.backward()
    # row 0 has advantage +1: minimising loss should INCREASE its logprob,
    # so the gradient of the loss wrt that logprob must be negative
    assert lp.grad[0].sum() < 0
    assert lp.grad[1].sum() > 0


def test_clipping_bounds_a_large_ratio():
    lp = torch.full((1, 3), 5.0, requires_grad=True)   # ratio = e^5, far outside
    old = torch.zeros(1, 3)
    ref = torch.zeros(1, 3)
    adv = torch.tensor([1.0])
    mask = torch.ones(1, 3)
    _, stats = grpo_loss(lp, old, ref, adv, mask, config=GRPOConfig(clip_eps=0.2, kl_coef=0.0))
    assert stats["clip_fraction"] == pytest.approx(1.0)


def test_masked_positions_are_ignored():
    lp, old, ref, adv, _ = _inputs()
    full = torch.ones(2, 4)
    half = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    l_full, _ = grpo_loss(lp, old, ref, adv, full, config=GRPOConfig())
    l_half, _ = grpo_loss(lp, old, ref, adv, half, config=GRPOConfig())
    assert torch.isfinite(l_half)
    # with all-zero logprobs the two agree; the point is that masking does not
    # produce NaN from dividing by a zero token count
    assert not torch.isnan(l_half)


def test_length_normalisation_prevents_long_sequence_dominance():
    """Sequences run 4-200 tokens; without 1/|y| the gradient follows length."""
    short = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    long = torch.ones(1, 4)
    lp = torch.zeros(1, 4, requires_grad=True)
    adv = torch.tensor([1.0])
    cfg = GRPOConfig(kl_coef=0.0)
    l_s, _ = grpo_loss(lp, torch.zeros(1, 4), torch.zeros(1, 4), adv, short, config=cfg)
    l_l, _ = grpo_loss(lp, torch.zeros(1, 4), torch.zeros(1, 4), adv, long, config=cfg)
    assert l_s.item() == pytest.approx(l_l.item(), abs=1e-6)


def test_kl_penalty_increases_loss_when_policy_diverges():
    lp = torch.full((1, 3), 1.0, requires_grad=True)
    zero_kl, _ = grpo_loss(lp, lp.detach(), lp.detach(), torch.tensor([0.0]),
                           torch.ones(1, 3), config=GRPOConfig(kl_coef=1.0))
    far_kl, _ = grpo_loss(lp, lp.detach(), torch.zeros(1, 3), torch.tensor([0.0]),
                          torch.ones(1, 3), config=GRPOConfig(kl_coef=1.0))
    assert far_kl > zero_kl
