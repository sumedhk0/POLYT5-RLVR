"""Novelty term: is this polymer absent from the reference corpus?

Wraps whatever index object is injected. The caller passes canonical PSMILES
so the index never pays canonicalization per query - measured at 253x the
cost of the lookup itself.
"""

from __future__ import annotations

from typing import Any

from polyt5.rewards.base import RewardResult


def novelty_reward(canonical_psmiles: str | None, index: Any | None) -> RewardResult:
    """Return 1.0 when the polymer is absent from the index.

    Args:
        canonical_psmiles: Canonical PSMILES, or None when decoding failed.
        index: Anything exposing ``is_novel(str) -> bool``. When None, novelty
            is reported as 0.0 rather than 1.0 - a missing reference set must
            never silently inflate a reward.

    Returns:
        RewardResult with a single ``novelty`` component.
    """
    if index is None or canonical_psmiles is None:
        return RewardResult(0.0, {"novelty": 0.0})
    try:
        novel = bool(index.is_novel(canonical_psmiles))
    except Exception:
        return RewardResult(0.0, {"novelty": 0.0})
    return RewardResult(float(novel), {"novelty": float(novel)})
