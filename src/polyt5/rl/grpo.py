"""The GRPO objective: clipped surrogate plus a KL anchor.

Clipping exists because the candidates were sampled under the old policy; as
theta moves, that data becomes off-policy, and an unclipped ratio times a large
advantage produces a destructive update.

Length normalisation is not optional here. Our sequences run 4-200 tokens and
mean length already varies 56-100 across sampling configurations, so without
the 1/|y| factor the gradient is dominated by long sequences and the policy
learns to pad.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GRPOConfig:
    """Objective hyperparameters.

    Attributes:
        clip_eps: Ratio clipping half-width.
        kl_coef: Weight on the KL anchor to the frozen reference policy. The
            primary knob: too low and the policy finds degenerate strings that
            score well; too high and it cannot move.
    """

    clip_eps: float = 0.2
    kl_coef: float = 0.02

    def __post_init__(self) -> None:
        if self.clip_eps <= 0:
            raise ValueError(f"clip_eps must be positive, got {self.clip_eps}")
        if self.kl_coef < 0:
            raise ValueError(f"kl_coef must be non-negative, got {self.kl_coef}")


#: Shared default config. Safe to reuse (frozen, holds only immutable floats),
#: and avoids a mutable-default-via-function-call in grpo_loss's signature.
_DEFAULT_CONFIG = GRPOConfig()


def k3_kl(logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
    """Unbiased, always-positive KL estimator used by GRPO.

    k3(x) = r - log r - 1 where r = pi_ref / pi_theta. Lower variance than the
    naive log-ratio estimator, and never negative, which makes it safe to add
    directly to a loss.

    Args:
        logprobs: log pi_theta for the chosen tokens.
        ref_logprobs: log pi_ref for the same tokens.

    Returns:
        Per-element KL estimate, same shape as the inputs.
    """
    log_ratio = (ref_logprobs - logprobs).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    return ratio - log_ratio - 1.0


def grpo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    config: GRPOConfig = _DEFAULT_CONFIG,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped surrogate plus KL, length-normalised per sequence.

    Args:
        logprobs: (B, T) log pi_theta of the sampled tokens, with gradient.
        old_logprobs: (B, T) log pi_theta_old, detached.
        ref_logprobs: (B, T) log pi_ref, detached.
        advantages: (B,) group-relative advantage per sequence.
        mask: (B, T) 1.0 on real tokens, 0.0 on padding.
        config: Clipping and KL settings.

    Returns:
        ``(loss, stats)`` where stats carries policy loss, KL, clip fraction
        and mean ratio for logging.
    """
    mask = mask.to(logprobs.dtype)
    token_counts = mask.sum(dim=1).clamp(min=1.0)

    log_ratio = (logprobs - old_logprobs).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    adv = advantages.unsqueeze(1)

    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv
    per_token = -torch.min(unclipped, clipped)
    policy_loss = ((per_token * mask).sum(dim=1) / token_counts).mean()

    kl_per_token = k3_kl(logprobs, ref_logprobs)
    kl = ((kl_per_token * mask).sum(dim=1) / token_counts).mean()

    loss = policy_loss + config.kl_coef * kl

    with torch.no_grad():
        was_clipped = ((ratio < 1.0 - config.clip_eps) | (ratio > 1.0 + config.clip_eps))
        clip_fraction = ((was_clipped.to(mask.dtype) * mask).sum() / mask.sum().clamp(min=1.0))
        stats = {
            "policy_loss": float(policy_loss.detach()),
            "kl": float(kl.detach()),
            "clip_fraction": float(clip_fraction),
            "mean_ratio": float(((ratio * mask).sum() / mask.sum().clamp(min=1.0)).detach()),
        }
    return loss, stats
