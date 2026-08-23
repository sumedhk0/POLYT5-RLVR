"""The four real arms, plus a negative control. Weights live in config, never
in code.

Every real arm rejects a structure that RDKit cannot parse before scoring
anything else: a structure that is not a polymer earns nothing on any axis.
Three arms (accuracy, composite, constraint) apply the combined SV+PV
validity gate up front and score the rest of their terms only on survivors.
``ValidityArm`` is the exception: its whole job is to *measure* the paper's
nested SV -> TSD -> DD -> PV cascade, so it checks SV first but defers PV
until after TSD and DD, in the cascade's own order - see its docstring.

``ControlArm`` is not one of the four: it is the study's negative control,
scoring every candidate with reward drawn uniformly at random from
``[0, 1)``, independent of the candidate's content entirely - no gate, no
chemistry, nothing read from ``pselfies`` at all. It exists to separate "this
specific reward design changed the metrics" from "training against ANY
reward signal changes the metrics, through RL mechanics alone" (entropy
change, KL drift, sampling shift). See its own docstring for the full
rationale and the three non-negotiable design points.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

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
    """Shared plumbing: gate, decode once, reuse the canonical form.

    ``ensemble_size`` is the number of members of the property predictor whose
    ``(mean, std, n_contributing_members)`` triples this arm will be fed. It
    has no default that guesses: a candidate only ONE member of a four-member
    ensemble could score reports ``std = 0.0`` exactly like a candidate all
    four agreed on, and only ``n_contributing / ensemble_size`` tells the two
    apart (see :mod:`polyt5.rewards.tg`). ``1`` -- the default -- means "a
    single-model predictor", which is the only configuration in which a
    one-member answer is genuinely full coverage; feeding a multi-member
    ensemble's triples to an arm still declaring ``1`` raises rather than
    silently restoring the inverted weight.
    """

    def __init__(self, *, novelty_index: Any | None = None, tolerance: float = 50.0,
                 sa_max: float = 6.0, tg_config: TgRewardConfig | None = None,
                 ensemble_size: int = 1) -> None:
        if ensemble_size < 1:
            raise ValueError(f"ensemble_size must be >= 1, got {ensemble_size}")
        self.novelty_index = novelty_index
        self.tolerance = tolerance
        self.sa_max = sa_max
        self.tg_config = tg_config or TgRewardConfig()
        self.ensemble_size = ensemble_size

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
        for pselfies, target, (mean, std, n) in zip(candidates, targets, predictions,
                                                    strict=True):
            gate, _ = self._prepare(pselfies)
            if gate.gated:
                out.append(gate)
                continue
            out.append(tg_reward(mean, std, target, n_contributing=n,
                                 n_total=self.ensemble_size, config=self.tg_config))
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
        # C2's reward reads neither `targets` nor `predictions` -- validity is
        # a fact about the structure alone. They are still zipped with
        # strict=True so a caller handing three misaligned sequences gets a
        # ValueError here, exactly as it would from every other arm, instead
        # of silence (the other three zip strictly at their own loops).
        for pselfies, _target, _prediction in zip(candidates, targets, predictions,
                                                  strict=True):
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
    """C3: weighted sum of the confidence-weighted Tg term and novelty, plus a
    flat bonus for having cleared the structural gate.

    The ``pv`` term is the literal constant ``weights["pv"] * 1.0``: every
    candidate that reaches this line has already passed the SV+PV gate in
    :meth:`_BaseArm._prepare`, and every candidate that has not returned its
    gated result instead. It therefore separates gated from non-gated
    candidates and NOTHING else -- within a GRPO group whose members all
    cleared the gate it is a constant that
    :func:`~polyt5.rl.advantages.group_advantages` removes entirely by
    mean-centring. Calling this "a weighted sum of accuracy, PV pass and
    novelty" oversold it; only two of the three terms can vary between two
    gate survivors.
    """

    def __init__(self, *, weights: dict[str, float] | None = None, **kw) -> None:
        super().__init__(**kw)
        self.weights = dict(weights or DEFAULT_COMPOSITE_WEIGHTS)

    def __call__(self, candidates, targets, predictions):
        out: list[RewardResult] = []
        for pselfies, target, (mean, std, n) in zip(candidates, targets, predictions,
                                                    strict=True):
            gate, canon = self._prepare(pselfies)
            if gate.gated:
                out.append(gate)
                continue
            tg = tg_reward(mean, std, target, n_contributing=n,
                           n_total=self.ensemble_size, config=self.tg_config)
            nov = novelty_reward(canon, self.novelty_index)
            value = (self.weights.get("tg", 0.0) * tg.value
                     + self.weights.get("pv", 0.0) * 1.0
                     + self.weights.get("novelty", 0.0) * nov.value)
            out.append(RewardResult(value, {**tg.components, **nov.components, "pv": 1.0}))
        return out


class ConstraintArm(_BaseArm):
    """C4: Tg window AND synthesisable AND novel, as a conjunction.

    C4 carries no continuous confidence weight -- that is the point of the
    arm, per spec section 4.3 -- but it must still not read a single member's
    guess as an ensemble consensus. The coverage of the property ensemble
    therefore enters as a fourth CONJUNCT (``ensemble_backed``, driven by
    :attr:`~polyt5.rewards.tg.TgRewardConfig.min_coverage`) rather than as a
    discount on the value. A single-model predictor reports ``n = 1`` of
    ``ensemble_size = 1``, i.e. coverage 1.0, and is unaffected.
    """

    def __call__(self, candidates, targets, predictions):
        out: list[RewardResult] = []
        for pselfies, target, (mean, _std, n) in zip(candidates, targets, predictions,
                                                     strict=True):
            gate, canon = self._prepare(pselfies)
            if gate.gated:
                out.append(gate)
                continue
            if not 0 <= n <= self.ensemble_size:
                raise ValueError(
                    f"n_contributing ({n}) must be in [0, ensemble_size] "
                    f"(ensemble_size={self.ensemble_size}): this arm's declared ensemble_size "
                    "does not match the predictor supplying these predictions."
                )
            coverage = n / self.ensemble_size
            sa = synthetic_accessibility(canon) if canon else None
            novel = bool(novelty_reward(canon, self.novelty_index).value)
            result = constraint_reward(
                abs(mean - target), sa, novel,
                tolerance=self.tolerance, sa_max=self.sa_max,
                ensemble_backed=coverage >= self.tg_config.min_coverage,
            )
            out.append(RewardResult(
                result.value,
                {**result.components, "coverage": coverage, "n_contributing": float(n)},
                result.gated,
                result.reason,
            ))
        return out


class ControlArm(_BaseArm):
    """Negative control: reward is uniform random in ``[0, 1)``, independent
    of the candidate entirely.

    Purpose
    -------
    Every one of the four real arms' whole premise is "this specific reward
    design changed what the policy learns to do". Nothing in a single GRPO
    run against a single accuracy/validity/composite/constraint arm can
    distinguish that claim from the alternative explanation that ANY reward
    signal - however uninformative - moves the metrics
    ``scripts/compare_arms.py`` reports (PV rate, diversity/duplicate rate,
    property MAE, ...), purely through the mechanics of RL itself: entropy
    change under repeated sampling, KL drift away from the frozen reference,
    or a sampling-distribution shift that has nothing to do with what the
    reward actually measures. This arm is the control for that confound: it
    is trained through the exact same :class:`~polyt5.rl.trainer.GRPOTrainer`
    loop as every other arm, on noise. If ``control``'s trained policy shows
    apparent improvement on ANY reported metric, that is direct evidence that
    an improvement in one of the other four arms is not, by itself,
    attributable to that arm's reward design - it would need to be shown to
    exceed what optimizing pure noise already produces. This arm has no
    "metric it optimized": ``scripts/compare_arms.py``'s ``ARM_METRIC`` has
    no entry for ``"control"``, so its row's ``success`` column is always
    ``None`` - it did not win or lose, because it was not competing.

    Three design points, none negotiable
    -------------------------------------
    * **Independent of the candidate's content.** Every other arm's
      ``__call__`` starts by running :meth:`_BaseArm._prepare` (the SV+PV
      gate) or :func:`~polyt5.chemistry.validity.validate_pselfies` directly.
      This one does neither - it never reads ``pselfies`` at all, valid or
      not. Gating invalid candidates to ``0.0`` here would make this
      secretly a second validity arm, correlated with PV by construction,
      not a control.
    * **Reproducible, not merely random.** ``np.random.default_rng([seed,
      step_index])`` - the SAME two-integer seeding
      :meth:`~polyt5.rl.trainer.GRPOTrainer.step` uses for its own target
      sampling (see that module's docstring) - so replaying step
      ``step_index`` of a run with the same ``seed`` reproduces the exact
      same rewards. The GLOBAL numpy RNG (``np.random.seed``/``np.random.
      rand``) is never touched, so these draws cannot be perturbed by, or
      accidentally perturb, anything else in the process that happens to use
      the global generator.
    * **Random, never constant.** A constant reward (e.g. always ``0.5``)
      gives every group zero within-group variance, hence an advantage of
      exactly ``0.0`` for every member (see
      :func:`~polyt5.rl.advantages.group_advantages`) and therefore NO
      policy gradient at all - only KL-anchored drift toward the reference.
      That would test nothing: an inert arm cannot show whether RL mechanics
      alone move the metrics, because RL's optimization step never actually
      ran on anything. Uniform-random reward keeps real within-group
      variance and therefore a real (if uninformative) gradient - the actual
      control.

    ``step_index`` is not part of the shared :class:`ArmReward` three-argument
    contract - it is an extra, keyword-only argument this arm alone accepts.
    :attr:`wants_step_index` is the duck-typed marker
    :meth:`~polyt5.rl.trainer.GRPOTrainer.step` reads to know to pass it
    through; every other arm's reward is a pure function of ``(candidates,
    targets, predictions)`` and does not declare it. Called without
    ``step_index`` (e.g. directly from a test, or from
    ``scripts/compare_arms.py``, which has no training step index at all) it
    defaults to ``0``.
    """

    #: Read by GRPOTrainer.step to decide whether to pass this call's
    #: step_index through. False (the class default, via getattr) for every
    #: other arm.
    wants_step_index = True

    def __init__(self, *, seed: int = 0, **kw: Any) -> None:
        super().__init__(**kw)
        self.seed = int(seed)

    def __call__(self, candidates, targets, predictions, *, step_index: int = 0):
        # Zipped purely to validate alignment, matching every other arm's
        # contract (e.g. ValidityArm.__call__) -- this arm reads none of the
        # three sequences' actual content, only their common length.
        for _ in zip(candidates, targets, predictions, strict=True):
            pass
        rng = np.random.default_rng([self.seed, int(step_index)])
        draws = rng.uniform(0.0, 1.0, size=len(candidates))
        return [RewardResult(float(value), {"control": float(value)}) for value in draws]


_ARMS = {"accuracy": AccuracyArm, "validity": ValidityArm,
         "composite": CompositeArm, "constraint": ConstraintArm,
         "control": ControlArm}


def build_arm(name: str, **kwargs: Any) -> ArmReward:
    """Construct an arm by name.

    Args:
        name: One of ``accuracy``, ``validity``, ``composite``, ``constraint``,
            ``control``.
        **kwargs: Passed to the arm - ``novelty_index``, ``tolerance``,
            ``sa_max``, ``tg_config``, ``ensemble_size``, and for composite,
            ``weights``. Any arm reading the Tg term MUST be given the
            ``ensemble_size`` of the predictor whose triples it will receive;
            see :class:`_BaseArm`. ``control`` additionally reads ``seed``
            (default ``0``) - see :class:`ControlArm`.

    Raises:
        ValueError: On an unknown arm name.
    """
    if name not in _ARMS:
        raise ValueError(f"unknown arm {name!r}; expected one of {sorted(_ARMS)}")
    return _ARMS[name](**kwargs)
