"""The four arms. Weights live in config, never in code.

Every arm applies the validity gate first: a structure that is not a polymer
earns nothing on any axis.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from polyt5.chemistry.canonicalization import canonical_psmiles
from polyt5.chemistry.conversion import pselfies_to_psmiles
from polyt5.chemistry.metrics import synthetic_accessibility
from polyt5.rewards.base import RewardResult
from polyt5.rewards.constraints import constraint_reward
from polyt5.rewards.novelty import novelty_reward
from polyt5.rewards.tg import TgRewardConfig, tg_reward
from polyt5.rewards.validity import validity_gate

#: (mean, std, n_contributing_members) as returned by the ensemble predictor.
Prediction = tuple[float, float, int]

DEFAULT_COMPOSITE_WEIGHTS = {"tg": 1.0, "pv": 0.5, "novelty": 0.25}


class ArmReward(Protocol):
    """Scores a batch of candidates for one arm."""

    def __call__(
        self,
        candidates: Sequence[str],
        targets: Sequence[float],
        predictions: Sequence[Prediction],
    ) -> list[RewardResult]:
        ...


class _BaseArm:
    """Shared plumbing: gate, decode once, reuse the canonical form."""

    def __init__(self, *, novelty_index: Any | None = None, tolerance: float = 50.0,
                 sa_max: float = 6.0, tg_config: TgRewardConfig | None = None) -> None:
        self.novelty_index = novelty_index
        self.tolerance = tolerance
        self.sa_max = sa_max
        self.tg_config = tg_config or TgRewardConfig()

    def _prepare(self, pselfies: str) -> tuple[RewardResult, str | None]:
        gate = validity_gate(pselfies)
        if gate.gated:
            return gate, None
        psmiles = pselfies_to_psmiles(pselfies)
        return gate, canonical_psmiles(psmiles) if psmiles else None


class AccuracyArm(_BaseArm):
    """C1: closeness to the requested Tg, discounted by ensemble disagreement."""

    def __call__(self, candidates, targets, predictions):
        out: list[RewardResult] = []
        for pselfies, target, (mean, std, _n) in zip(candidates, targets, predictions,
                                                     strict=True):
            gate, _ = self._prepare(pselfies)
            if gate.gated:
                out.append(gate)
                continue
            out.append(tg_reward(mean, std, target, config=self.tg_config))
        return out


class ValidityArm(_BaseArm):
    """C2: binary - did the candidate survive the full filter cascade?"""

    def __call__(self, candidates, targets, predictions):
        out: list[RewardResult] = []
        for pselfies in candidates:
            gate, canon = self._prepare(pselfies)
            if gate.gated:
                out.append(gate)
                continue
            novel = novelty_reward(canon, self.novelty_index).value
            out.append(RewardResult(1.0, {"pv": 1.0, "novelty": novel}))
        return out


class CompositeArm(_BaseArm):
    """C3: weighted sum of accuracy, PV pass, and novelty."""

    def __init__(self, *, weights: dict[str, float] | None = None, **kw) -> None:
        super().__init__(**kw)
        self.weights = dict(weights or DEFAULT_COMPOSITE_WEIGHTS)

    def __call__(self, candidates, targets, predictions):
        out: list[RewardResult] = []
        for pselfies, target, (mean, std, _n) in zip(candidates, targets, predictions,
                                                     strict=True):
            gate, canon = self._prepare(pselfies)
            if gate.gated:
                out.append(gate)
                continue
            tg = tg_reward(mean, std, target, config=self.tg_config)
            nov = novelty_reward(canon, self.novelty_index)
            value = (self.weights.get("tg", 0.0) * tg.value
                     + self.weights.get("pv", 0.0) * 1.0
                     + self.weights.get("novelty", 0.0) * nov.value)
            out.append(RewardResult(value, {**tg.components, **nov.components, "pv": 1.0}))
        return out


class ConstraintArm(_BaseArm):
    """C4: Tg window AND synthesisable AND novel, as a conjunction."""

    def __call__(self, candidates, targets, predictions):
        out: list[RewardResult] = []
        for pselfies, target, (mean, _std, _n) in zip(candidates, targets, predictions,
                                                      strict=True):
            gate, canon = self._prepare(pselfies)
            if gate.gated:
                out.append(gate)
                continue
            sa = synthetic_accessibility(canon) if canon else None
            novel = bool(novelty_reward(canon, self.novelty_index).value)
            out.append(constraint_reward(abs(mean - target), sa, novel,
                                         tolerance=self.tolerance, sa_max=self.sa_max))
        return out


_ARMS = {"accuracy": AccuracyArm, "validity": ValidityArm,
         "composite": CompositeArm, "constraint": ConstraintArm}


def build_arm(name: str, **kwargs: Any) -> ArmReward:
    """Construct an arm by name.

    Args:
        name: One of ``accuracy``, ``validity``, ``composite``, ``constraint``.
        **kwargs: Passed to the arm - ``novelty_index``, ``tolerance``,
            ``sa_max``, ``tg_config``, and for composite, ``weights``.

    Raises:
        ValueError: On an unknown arm name.
    """
    if name not in _ARMS:
        raise ValueError(f"unknown arm {name!r}; expected one of {sorted(_ARMS)}")
    return _ARMS[name](**kwargs)
