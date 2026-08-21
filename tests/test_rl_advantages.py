from __future__ import annotations

import numpy as np
import pytest

from polyt5.rl.advantages import group_advantages


def test_advantage_is_group_relative_not_global():
    # two groups of 4; group 2 has uniformly higher rewards
    rewards = np.array([0.0, 1.0, 0.0, 1.0, 10.0, 11.0, 10.0, 11.0])
    adv = group_advantages(rewards, group_size=4)
    # within each group the pattern is identical, because the baseline is the
    # group's own mean - this is the whole point of GRPO
    assert np.allclose(adv[:4], adv[4:])


def test_advantages_are_zero_mean_within_each_group():
    rng = np.random.default_rng(0)
    rewards = rng.normal(size=12)
    adv = group_advantages(rewards, group_size=4)
    for g in range(3):
        assert adv[g * 4:(g + 1) * 4].mean() == pytest.approx(0.0, abs=1e-6)


def test_identical_rewards_give_zero_advantage_not_nan():
    """The degenerate case: every candidate in a group scored the same."""
    adv = group_advantages(np.array([0.5, 0.5, 0.5, 0.5]), group_size=4)
    assert np.all(np.isfinite(adv))
    assert np.allclose(adv, 0.0)


def test_better_than_average_is_positive():
    adv = group_advantages(np.array([0.0, 0.0, 0.0, 1.0]), group_size=4)
    assert adv[3] > 0 and np.all(adv[:3] < 0)


def test_rejects_ragged_input():
    with pytest.raises(ValueError, match="multiple"):
        group_advantages(np.zeros(7), group_size=4)


def test_advantages_are_scaled_by_group_spread():
    """A mean-centring-only implementation passes every other test in this file.

    Groups [0,1,0,1] and [0,10,0,10] have the same shape but 10x different
    spread. Dividing by the group standard deviation maps both to the same
    +/-1 magnitudes; merely subtracting the group mean would leave them 10x
    apart. This is what makes these advantages rather than raw deviations.
    """
    narrow = group_advantages(np.array([0.0, 1.0, 0.0, 1.0]), group_size=4)
    wide = group_advantages(np.array([0.0, 10.0, 0.0, 10.0]), group_size=4)
    assert np.allclose(narrow, wide, atol=1e-6)
    assert np.allclose(np.abs(narrow), 1.0, atol=1e-6)


def test_rejects_non_positive_group_size():
    with pytest.raises(ValueError, match="group_size"):
        group_advantages(np.zeros(4), group_size=0)
