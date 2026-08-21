# tests/test_rl_grpo.py
from __future__ import annotations

import math

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


def test_k3_kl_is_non_negative_at_and_beyond_the_clamp_boundary():
    """log_ratio is clamped to +/-20 before both exp() and the linear term.

    A mutant that clamps only the exponential while leaving the raw,
    unclamped log_ratio in the linear term stays positive until the raw
    value exceeds exp(20) ~= 4.85e8 -- past that, ratio - log_ratio - 1
    goes NEGATIVE because the (capped) exponential term can no longer
    outrun the (uncapped) linear term. -25/25 sit just past the +/-20
    clamp edge and stay positive either way, so they're kept as ordinary
    regression coverage; 6e8/1e9 are the values that actually discriminate
    the bug.
    """
    for delta in (-25.0, 25.0, 6e8, 1e9):
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


def test_clipping_changes_the_loss_value_not_just_the_statistic():
    """clip_fraction is computed alongside the loss, not from it.

    A mutant with clipping stripped from the loss path still reports
    clip_fraction == 1.0, so only the loss VALUE distinguishes the two.
    ratio = exp(5) = 148.41; clipped to 1.2 the loss is -1.20, unclipped
    it would be -148.41.
    """
    lp = torch.full((1, 3), 5.0, requires_grad=True)
    zeros = torch.zeros(1, 3)
    mask = torch.ones(1, 3)
    cfg = GRPOConfig(clip_eps=0.2, kl_coef=0.0)

    loss_pos, _ = grpo_loss(lp, zeros, zeros, torch.tensor([1.0]), mask, config=cfg)
    assert loss_pos.item() == pytest.approx(-1.2, abs=1e-4)

    # negative advantage: min() selects the UNCLIPPED term by design
    loss_neg, _ = grpo_loss(lp, zeros, zeros, torch.tensor([-1.0]), mask, config=cfg)
    assert loss_neg.item() == pytest.approx(148.4132, rel=1e-4)


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


def test_all_padding_row_does_not_produce_nan():
    """A group can contain a fully-padded row (e.g. group_size > number of
    real candidates); the token-count clamp exists for exactly this case.
    Without it, an all-zero mask row divides 0/0 in that row's own
    normalisation, and the resulting NaN survives the batch mean and
    poisons the whole loss -- even though row 0 is entirely normal.
    """
    lp = torch.zeros(2, 4, requires_grad=True)
    old = torch.zeros(2, 4)
    ref = torch.zeros(2, 4)
    adv = torch.tensor([1.0, -1.0])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    loss, stats = grpo_loss(lp, old, ref, adv, mask, config=GRPOConfig())
    assert torch.isfinite(loss)
    for value in stats.values():
        assert math.isfinite(value)
    # row 0 (all real, ratio == 1, advantage +1) is unaffected by row 1's
    # padding: policy_loss == mean(-1.0, 0.0) == -0.5, and with identical
    # logprobs/ref the KL term is exactly zero, so loss == -0.5 exactly.
    assert loss.item() == pytest.approx(-0.5, abs=1e-6)
    loss.backward()
    assert lp.grad is not None and torch.isfinite(lp.grad).all()


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


def test_kl_coefficient_scales_the_kl_term_linearly():
    """With zero advantage the policy term vanishes entirely, so the loss
    IS exactly kl_coef * kl. A mutant that adds the KL term without the
    coefficient (e.g. `loss = policy_loss + kl`) is indistinguishable from
    a correct implementation at kl_coef=1.0 alone -- this pins an
    intermediate coefficient where a dropped multiply becomes visible.

    logprobs=1.0, ref_logprobs=0.0 -> log_ratio = ref - logprobs = -1.0,
    k3 = exp(-1) - (-1) - 1 = exp(-1) ~= 0.367879.
    """
    lp = torch.full((1, 3), 1.0, requires_grad=True)
    old = torch.zeros(1, 3)
    ref = torch.zeros(1, 3)
    adv = torch.tensor([0.0])
    mask = torch.ones(1, 3)
    expected_k3 = math.exp(-1.0)

    loss_half, _ = grpo_loss(lp, old, ref, adv, mask, config=GRPOConfig(kl_coef=0.5))
    loss_full, _ = grpo_loss(lp, old, ref, adv, mask, config=GRPOConfig(kl_coef=1.0))

    assert loss_full.item() == pytest.approx(expected_k3, rel=1e-4)
    assert loss_half.item() == pytest.approx(0.5 * expected_k3, rel=1e-4)
    assert loss_half.item() == pytest.approx(0.5 * loss_full.item(), rel=1e-4)
