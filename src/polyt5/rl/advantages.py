"""Group-relative advantage - the piece that lets GRPO drop the critic.

PPO learns a value network to supply a baseline. GRPO instead samples a group
of candidates per prompt and uses the group's own mean. For a 7.5M-parameter
policy on a 12 GB card, not carrying a second network of comparable size is a
material saving, and the critic is the usual source of PPO instability.

It also suits a noisy reward: our Tg predictor has 28.7 K held-out MAE, and
normalising within a group cancels per-prompt bias - a predictor that reads
systematically high at 500 K shifts every member of that group equally and
leaves the advantage untouched.
"""

from __future__ import annotations

import numpy as np


def group_advantages(rewards: np.ndarray, group_size: int, *, eps: float = 1e-8) -> np.ndarray:
    """Normalise rewards within each contiguous group.

    Args:
        rewards: Flat array, groups laid out contiguously.
        group_size: Candidates per prompt.
        eps: Added to the standard deviation to avoid dividing by zero.

    Returns:
        Advantages, same shape as ``rewards``. A group whose members all scored
        identically yields exactly zero - no gradient, rather than NaN.

    Raises:
        ValueError: If the length is not a multiple of ``group_size``.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    if rewards.size % group_size:
        raise ValueError(
            f"rewards length {rewards.size} is not a multiple of group_size {group_size}"
        )
    grouped = rewards.reshape(-1, group_size)
    centred = grouped - grouped.mean(axis=1, keepdims=True)
    std = grouped.std(axis=1, keepdims=True)
    return (centred / (std + eps)).reshape(-1)
