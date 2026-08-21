"""The four arms. Weights live in config, never in code.

Every arm rejects a structure that RDKit cannot parse before scoring anything
else: a structure that is not a polymer earns nothing on any axis. Three arms
(accuracy, composite, constraint) apply the combined SV+PV validity gate up
front and score the rest of their terms only on survivors. ``ValidityArm`` is
the exception: its whole job is to *measure* the paper's nested SV -> TSD ->
DD -> PV cascade, so it checks SV first but defers PV until after TSD and DD,
in the cascade's own order - see its docstring.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from polyt5.chemistry.canonicalization import canonical_psmiles
from polyt5.chemistry.conversion import pselfies_to_psmiles
from polyt5.chemistry.metrics import synthetic_accessibility
from polyt5.chemistry.validity import validate_pselfies
from polyt5.evaluation import has_valid_termini
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
    """C2: R = 1 only if the candidate clears the nested SV -> TSD -> DD -> PV
    cascade; 0 otherwise.

    Each stage runs only if every earlier stage passed:

    * **SV** - RDKit parses and sanitizes the structure.
    * **TSD** - absent from the injected reference index; a training-set
      duplicate fails here even though it is a well-formed polymer.
    * **DD** - first occurrence of its canonical form within this call's
      batch; a later duplicate fails here even though the first copy passed.
    * **PV** - exactly the expected number of termini, each with valency one.

    ``components`` records the boolean outcome of every stage the candidate
    actually reached (``sv``, ``tsd``, ``dd``, ``pv``); a stage the candidate
    never got to keeps its default of 0.0, so the first-failed stage is always
    identifiable. This mirrors :func:`polyt5.evaluation.apply_filter_cascade`,
    adapted to the injected-index protocol (``is_novel``) reward workers use.

    ``gated`` is reserved for structural failure (SV or PV) - the meaning it
    carries everywhere else in this package (:func:`polyt5.rewards.validity.
    validity_gate`, and every other arm's ``_prepare``), and what Task 7's
    trainer reads as ``gated_fraction`` to diagnose how much of a rollout
    batch is chemically invalid. A TSD or DD failure is perfectly valid
    chemistry that simply misses this arm's reward, not structural invalidity,
    so it sets ``value=0.0`` with ``gated=False`` and ``reason=None`` - the
    same convention :class:`ConstraintArm` already uses for its own
    conjunction failures - and the stage stays visible in ``components``.

    Without an injected novelty index the TSD stage can never be evaluated,
    and this package's fail-closed rule (a missing capability must never
    inflate a reward) would then zero every candidate silently: every GRPO
    group would have zero reward variance and the policy would receive no
    gradient at all, with no error raised. The constructor therefore requires
    ``novelty_index`` by default; pass ``require_novelty_index=False`` to
    explicitly opt into treating a missing index as a TSD no-op (every SV
    survivor passes straight to DD), matching
    :func:`polyt5.evaluation.apply_filter_cascade`'s own behaviour when its
    ``training_index`` is ``None``.
    """

    def __init__(self, *, require_novelty_index: bool = True, **kw) -> None:
        super().__init__(**kw)
        if require_novelty_index and self.novelty_index is None:
            raise ValueError(
                "ValidityArm cannot evaluate the TSD stage without a novelty_index: with "
                "none, TSD fails closed for every candidate (this package's rule - a missing "
                "capability must never inflate a reward), so every candidate would score "
                "0.0, every GRPO group would have zero reward variance, and the policy would "
                "receive no gradient at all, with no error raised. Pass a novelty_index, or "
                "pass require_novelty_index=False to explicitly opt into treating TSD as a "
                "no-op (every SV survivor passes straight to DD)."
            )
        self.require_novelty_index = require_novelty_index

    def __call__(self, candidates, targets, predictions):
        out: list[RewardResult] = []
        seen_canonical: set[str] = set()
        tsd_is_noop = self.novelty_index is None and not self.require_novelty_index
        for pselfies in candidates:
            verdict = validate_pselfies(pselfies)
            components = {"sv": float(verdict.valid), "tsd": 0.0, "dd": 0.0, "pv": 0.0}

            if not verdict.valid:
                out.append(RewardResult(0.0, components, True, verdict.reason or "invalid"))
                continue

            canon = verdict.canonical_psmiles
            novel = True if tsd_is_noop else bool(
                novelty_reward(canon, self.novelty_index).value
            )
            components["tsd"] = float(novel)
            if not novel:
                out.append(RewardResult(0.0, components))
                continue

            first_occurrence = canon not in seen_canonical
            components["dd"] = float(first_occurrence)
            if not first_occurrence:
                out.append(RewardResult(0.0, components))
                continue
            seen_canonical.add(canon)

            pv = bool(verdict.correct_termini and has_valid_termini(canon))
            components["pv"] = float(pv)
            if not pv:
                reason = verdict.reason if not verdict.correct_termini else "terminus_valency"
                out.append(RewardResult(0.0, components, True, reason))
                continue

            out.append(RewardResult(1.0, components, False, None))
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
