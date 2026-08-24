"""Group A Task 5: masked mean pooling, the regression head, and the two losses."""

from __future__ import annotations

import math

import pytest
import torch

from polyt5.model.heads import (
    RegressionHead,
    masked_mean_pool,
    weighted_huber_loss,
    weighted_lm_loss,
)


def test_pooling_ignores_padded_positions():
    """Padding must not drag the pooled vector toward zero."""
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = masked_mean_pool(hidden, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 3.0]]))


def test_pooling_matches_a_plain_mean_when_nothing_is_padded():
    hidden = torch.randn(3, 7, 5)
    mask = torch.ones(3, 7, dtype=torch.long)
    assert torch.allclose(masked_mean_pool(hidden, mask), hidden.mean(dim=1), atol=1e-6)


def test_pooling_of_an_all_pad_row_does_not_divide_by_zero():
    hidden = torch.randn(1, 4, 3)
    pooled = masked_mean_pool(hidden, torch.zeros(1, 4, dtype=torch.long))
    assert torch.isfinite(pooled).all()
    assert torch.allclose(pooled, torch.zeros(1, 3))


def test_pooling_is_invariant_to_extra_padding():
    hidden = torch.randn(1, 4, 6)
    short = masked_mean_pool(hidden[:, :4], torch.tensor([[1, 1, 1, 1]]))
    padded_hidden = torch.cat([hidden, torch.randn(1, 3, 6)], dim=1)
    long = masked_mean_pool(padded_hidden, torch.tensor([[1, 1, 1, 1, 0, 0, 0]]))
    assert torch.allclose(short, long, atol=1e-6)


def test_pooling_within_a_mixed_padding_batch_is_still_masked_per_row():
    """A batch whose rows carry different amounts of padding: each row's pooled
    vector must depend only on its own unmasked tokens, not on how long its
    batch-mates are. A test where every row shares the same padding count
    cannot detect a bug that pools over the whole (fixed) sequence length
    instead of each row's own mask sum.
    """
    row0 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [99.0, 99.0], [99.0, 99.0]])
    row1 = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    hidden = torch.stack([row0, row1])
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    pooled = masked_mean_pool(hidden, mask)
    # row0 (2 real tokens): mean([1,3]), mean([2,4]) = 2, 3
    # row1 (4 real tokens): mean([5,7,9,11]), mean([6,8,10,12]) = 8, 9
    assert torch.allclose(pooled, torch.tensor([[2.0, 3.0], [8.0, 9.0]]))
    # and adding row1's padding did not disturb row0's own value
    row0_alone = masked_mean_pool(row0[:2].unsqueeze(0), torch.tensor([[1, 1]]))
    assert torch.allclose(pooled[0], row0_alone[0])


def test_regression_head_shapes():
    head = RegressionHead(16, 1, dropout=0.0)
    assert head.n_outputs == 1
    assert head(torch.randn(5, 16)).shape == (5, 1)
    wide = RegressionHead(16, 100, dropout=0.0)
    assert wide(torch.randn(5, 16)).shape == (5, 100)


def test_regression_head_rejects_degenerate_sizes():
    with pytest.raises(ValueError, match="d_model"):
        RegressionHead(0, 1)
    with pytest.raises(ValueError, match="n_outputs"):
        RegressionHead(16, 0)


def test_huber_is_quadratic_near_zero_and_linear_far_out():
    small = weighted_huber_loss(torch.tensor([0.1]), torch.tensor([0.0]), delta=1.0)
    assert small == pytest.approx(0.5 * 0.1**2)
    big = weighted_huber_loss(torch.tensor([100.0]), torch.tensor([0.0]), delta=1.0)
    assert big == pytest.approx(1.0 * (100.0 - 0.5))
    # the whole point of Huber over MSE: a 145 K outlier is not squared
    assert big < weighted_huber_loss(torch.tensor([200.0]), torch.tensor([0.0]), delta=1.0)


def test_uniform_weights_equal_no_weights():
    pred, target = torch.randn(8), torch.randn(8)
    assert weighted_huber_loss(pred, target) == pytest.approx(
        float(weighted_huber_loss(pred, target, weights=torch.ones(8))), abs=1e-6
    )


def test_a_zero_weight_example_does_not_contribute():
    pred = torch.tensor([0.0, 1000.0])
    target = torch.tensor([0.0, 0.0])
    loss = weighted_huber_loss(pred, target, weights=torch.tensor([1.0, 0.0]))
    assert loss == pytest.approx(0.0)


def test_huber_reduces_a_multi_output_target_per_example_first():
    pred = torch.zeros(2, 4)
    target = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    loss = weighted_huber_loss(pred, target, weights=torch.tensor([1.0, 0.0]))
    assert loss == pytest.approx(0.0), "per-example weights must apply after the column mean"


def test_huber_weighted_average_uses_the_weight_sum_as_denominator():
    """Non-uniform, non-zero weights with a hand-computed expected value.

    A zero-weight or all-equal-weight test cannot catch a normalization bug
    (e.g. dividing by batch size instead of the weight sum): both the correct
    and the buggy denominator agree in those degenerate cases. Here they do
    not.
    """
    pred = torch.tensor([0.0, 0.0])
    target = torch.tensor([0.0, 2.0])
    loss = weighted_huber_loss(pred, target, delta=1.0, weights=torch.tensor([1.0, 3.0]))
    # huber(0, 0, delta=1) = 0.5 * 0**2 = 0.0
    # huber(0, 2, delta=1) = 1.0 * (2.0 - 0.5) = 1.5
    # weighted average = (0.0 * 1 + 1.5 * 3) / (1 + 3) = 1.125
    assert loss == pytest.approx(1.125)


def test_weighted_lm_loss_matches_token_cross_entropy_on_equal_length_rows():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 7)
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
    reference = torch.nn.functional.cross_entropy(
        logits.view(-1, 7), labels.view(-1), ignore_index=-100
    )
    assert weighted_lm_loss(logits, labels) == pytest.approx(float(reference), abs=1e-6)


def test_weighted_lm_loss_ignores_padded_label_positions():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 7)
    padded = torch.tensor([[1, 2, -100, -100]])
    trimmed = torch.tensor([[1, 2]])
    assert weighted_lm_loss(logits, padded) == pytest.approx(
        float(weighted_lm_loss(logits[:, :2], trimmed)), abs=1e-6
    )


def test_weighted_lm_loss_honours_a_zero_weight_row():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 7)
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
    only_first = weighted_lm_loss(logits, labels, weights=torch.tensor([1.0, 0.0]))
    assert only_first == pytest.approx(
        float(weighted_lm_loss(logits[:1], labels[:1])), abs=1e-6
    )


def test_weighted_lm_loss_weighted_average_uses_the_weight_sum_as_denominator():
    """Non-uniform, non-zero weights on two rows with distinct per-example loss,
    checked against an independently hand-derived weighted average -- not
    against a second call to ``weighted_lm_loss`` itself. A wrong
    normalization denominator would not coincide with the correct one here
    the way it can when weights are uniform or zero (see the huber
    counterpart above for why).
    """
    # Row 0: near-certain correct prediction -> tiny cross-entropy.
    # Row 1: uniform logits -> cross-entropy is exactly log(2).
    logits = torch.tensor([[[10.0, 0.0]], [[0.0, 0.0]]])
    labels = torch.tensor([[0], [0]])
    weights = torch.tensor([1.0, 3.0])
    loss = weighted_lm_loss(logits, labels, weights=weights)
    ce0 = math.log1p(math.exp(-10.0))  # -log(sigmoid(10)) = log(1 + exp(-10))
    ce1 = math.log(2.0)
    expected = (ce0 * 1.0 + ce1 * 3.0) / (1.0 + 3.0)
    assert loss == pytest.approx(expected, abs=1e-6)
