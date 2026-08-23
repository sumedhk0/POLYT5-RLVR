"""The RLVR reward arms. Weights live in config, never in code.

[PRE-REGISTRATION NOTE, dated 2026-08-23] Tg was dropped from every RLVR
reward on this date. A generated polymer has no experimental Tg and never
will -- nobody has synthesised it -- so a Tg reward is a learned model's
opinion about the candidate, not a computed fact about it, and an arm built
on one is not RLVR (see :mod:`polyt5.rewards.tg`'s own warning, which predates
this change and is the reason for it). The new arm set is fully verifiable:
every reward below is either a deterministic structural check (RDKit parse,
terminus valency, an SA score against a fixed threshold) or a set-membership
test (novelty against the training corpus) -- never a model's prediction.

Two arms existed already and are affected differently by this cut:

* ``accuracy`` (C1) is **RETIRED, not deleted**. It trained to completion (a
  full round-1 run) BEFORE this change, and its result -- reward-scored error
  fell while diversity collapsed 0.951 -> 0.535 -- is kept as the motivating
  negative finding for dropping Tg everywhere else. Its class and config are
  unchanged and it stays registered under :func:`build_arm` so the completed
  run remains reproducible (e.g. via ``--resume``) and reportable by
  ``scripts/compare_arms.py``; it is simply no longer part of the arm set a
  NEW run should choose from.
* ``composite`` (C3) and ``constraint`` (C4) are **REDEFINED, not deleted**.
  Deleting them would leave only ``validity`` + ``control`` -- one objective
  and its control, with no off-diagonal at all. The off-diagonal (does
  optimising one verifiable axis damage another?) is the actual science here;
  it is what made ``accuracy``'s diversity collapse a finding rather than an
  anecdote. The redefinitions below preserve that question using only
  checkable quantities. **Neither arm has been trained under its new
  definition -- no data exists for either.** Redefining their rewards now is
  therefore a scope change made BEFORE any observation, not a criterion
  chosen after seeing results; see this same date's amendment in
  ``artifacts/baseline/frozen_baseline.json`` for the dated record of that
  distinction, and each redefined class's own docstring below.

Two new arms, ``novelty`` and ``synthesisability``, complete the set: each is
a single verifiable axis plus the validity gate, structurally mirroring how
``accuracy`` and ``constraint`` used to add exactly one axis on top of the
same gate.

Every real arm rejects a structure that RDKit cannot parse before scoring
anything else: a structure that is not a polymer earns nothing on any axis.
Every arm except ``ValidityArm`` and ``ControlArm`` applies the combined
SV+PV validity gate up front (:meth:`_BaseArm._prepare`) and scores the rest
of its terms only on survivors. ``ValidityArm`` is the exception: its whole
job is to *measure* the paper's nested SV -> TSD -> DD -> PV cascade, so it
checks SV first but defers PV until after TSD and DD, in the cascade's own
order - see its docstring.

``ControlArm`` is not one of the verifiable arms: it is the study's negative
control, scoring every candidate with reward drawn uniformly at random from
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
from polyt5.rewards.sa import sa_reward
from polyt5.rewards.tg import TgRewardConfig, tg_reward
from polyt5.rewards.validity import validity_gate

#: (mean, std, n_contributing_members) as returned by the ensemble predictor.
#: Retained for arms that still read it (``accuracy``, retired but kept
#: functional). Every other arm's ``ArmReward.__call__`` still accepts a
#: ``predictions`` sequence -- it is part of the shared three-argument
#: contract, and every arm still zips it against ``candidates`` for an
#: alignment check -- but does not read a single value out of it. See
#: ``tests/test_rewards.py``'s predictor-invariance tests.
Prediction = tuple[float, float, int]

#: ``CompositeArm``'s default weights, REDEFINED 2026-08-23 (see this
#: module's docstring): four fully verifiable terms -- PV (constant 1.0 for
#: every gate survivor), novelty, SA-pass, and within-batch diversity -- in
#: place of the old tg/pv/novelty mix. Equal weighting, not tuned: no data
#: exists yet for this arm under any weighting.
DEFAULT_COMPOSITE_WEIGHTS = {"pv": 0.25, "novelty": 0.25, "sa": 0.25, "diversity": 0.25}


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
    """C1: closeness to the requested Tg, discounted by ensemble disagreement.

    .. warning::

       **RETIRED, 2026-08-23. Not RLVR.** Tg has no ground truth for a
       generated candidate -- nobody has synthesised it -- so this reward is
       a learned model's opinion, not a computed fact; see
       :mod:`polyt5.rewards.tg`'s own warning. This arm is kept, unchanged,
       ONLY because it already trained to completion (round-1, 2000 steps)
       BEFORE the Tg cut, and its result stands as the motivating negative
       finding for making every other arm verifiable: reward-scored error
       fell while diversity collapsed 0.951 -> 0.535. It remains registered
       under :func:`build_arm` and its config (``configs/rl/accuracy.yaml``)
       is untouched so the completed run stays reproducible (e.g.
       ``train_grpo.py --resume``) and so ``scripts/compare_arms.py`` can
       still report its row. Do not choose this arm for a new run -- see the
       module docstring for the current, fully verifiable arm set.
    """

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


class NoveltyArm(_BaseArm):
    """R = 1.0 iff the candidate clears the SV+PV gate AND is absent from the
    training corpus, else 0.0.

    New arm (2026-08-23) -- see this module's docstring. Fully verifiable:
    novelty is set membership against the injected reference index
    (:func:`~polyt5.rewards.novelty.novelty_reward`), never a model's opinion.

    Unlike :class:`ValidityArm`'s TSD stage, a missing ``novelty_index`` here
    degrades to ``novel = 0.0`` (via :func:`~polyt5.rewards.novelty.
    novelty_reward`'s own None-handling) rather than raising at construction.
    That mirrors how the OLD ``CompositeArm``/``ConstraintArm`` always treated
    their novelty term, and keeps this arm safe to build under
    ``scripts/compare_arms.py --allow-missing-novelty-index`` (which passes
    ``novelty_index=None`` explicitly, by design) without a second opt-out
    kwarg to thread through ``scripts/train_grpo.py``'s ``build_reward_arm``.
    A silently-None index during actual TRAINING is not a realistic risk the
    way it is for ``ValidityArm``'s TSD stage: ``ARMS_NEEDING_NOVELTY_INDEX``
    always opens a real path for this arm, and a genuinely missing file
    raises in ``ScalableNoveltyIndex.open`` before an arm is ever built.
    """

    def __call__(self, candidates, targets, predictions):
        out: list[RewardResult] = []
        # This arm reads neither `targets` nor `predictions` -- novelty is a
        # fact about the structure alone, checked against an injected index.
        # Still zipped with strict=True so misaligned inputs raise here, like
        # every other arm.
        for pselfies, _target, _prediction in zip(candidates, targets, predictions,
                                                  strict=True):
            gate, canon = self._prepare(pselfies)
            if gate.gated:
                out.append(gate)
                continue
            novel = bool(novelty_reward(canon, self.novelty_index).value)
            out.append(RewardResult(float(novel), {"novel": float(novel)}))
        return out


class SynthesisabilityArm(_BaseArm):
    """R = 1.0 iff the candidate clears the SV+PV gate AND its RDKit SA score
    is <= ``sa_max``, else 0.0.

    New arm (2026-08-23) -- see this module's docstring. Fully verifiable:
    the SA score is a deterministic RDKit heuristic over the candidate's own
    structure (:func:`~polyt5.rewards.sa.sa_reward`), never a model's
    prediction and never conditioned on any target.
    """

    def __call__(self, candidates, targets, predictions):
        out: list[RewardResult] = []
        for pselfies, _target, _prediction in zip(candidates, targets, predictions,
                                                  strict=True):
            gate, canon = self._prepare(pselfies)
            if gate.gated:
                out.append(gate)
                continue
            out.append(sa_reward(canon, sa_max=self.sa_max))
        return out


class CompositeArm(_BaseArm):
    """C3, REDEFINED 2026-08-23 (see this module's docstring): weighted sum of
    four fully verifiable, structural terms -- PV, novelty, SA-pass, and
    within-batch diversity. No term reads a model prediction.

    Previously a weighted mix that included the Tg closeness term; that term
    is gone, not merely zero-weighted -- ``predictions`` is no longer read at
    all (see ``tests/test_rewards.py``'s predictor-invariance tests). **This
    arm has never been trained under this definition; no data exists for it
    under any weighting.**

    The ``pv`` term is the literal constant ``weights["pv"] * 1.0``: every
    candidate that reaches this line has already passed the SV+PV gate in
    :meth:`_BaseArm._prepare`, and every candidate that has not returned its
    gated result instead. It therefore separates gated from non-gated
    candidates and NOTHING else -- within a GRPO group whose members all
    cleared the gate it is a constant that
    :func:`~polyt5.rl.advantages.group_advantages` removes entirely by
    mean-centring.

    The ``diversity`` term is first-occurrence-within-this-batch, exactly
    :class:`ValidityArm`'s own DD stage: ``1.0`` the first time a candidate's
    canonical form appears in THIS ``__call__``'s batch, ``0.0`` for every
    later duplicate. This is ``docs/rlvr_plan.md``'s original diversity design
    ("penalize mode collapse onto a few polymers"), reused rather than
    reinvented, and -- like DD -- it is computed only over candidates that
    cleared the gate.
    """

    def __init__(self, *, weights: dict[str, float] | None = None, **kw) -> None:
        super().__init__(**kw)
        self.weights = dict(weights or DEFAULT_COMPOSITE_WEIGHTS)

    def __call__(self, candidates, targets, predictions):
        out: list[RewardResult] = []
        seen_canonical: set[str] = set()
        # `predictions` is part of the shared three-argument contract but is
        # never read -- this arm's whole point is that no term depends on a
        # model's prediction. See test_composite_arm_ignores_predictions.
        for pselfies, _target, _prediction in zip(candidates, targets, predictions,
                                                  strict=True):
            gate, canon = self._prepare(pselfies)
            if gate.gated:
                out.append(gate)
                continue
            novel = bool(novelty_reward(canon, self.novelty_index).value)
            sa = sa_reward(canon, sa_max=self.sa_max)
            first_occurrence = canon not in seen_canonical
            seen_canonical.add(canon)
            value = (self.weights.get("pv", 0.0) * 1.0
                     + self.weights.get("novelty", 0.0) * float(novel)
                     + self.weights.get("sa", 0.0) * sa.value
                     + self.weights.get("diversity", 0.0) * float(first_occurrence))
            out.append(RewardResult(value, {
                "pv": 1.0, "novelty": float(novel), **sa.components,
                "diversity": float(first_occurrence),
            }))
        return out


class ConstraintArm(_BaseArm):
    """C4, REDEFINED 2026-08-23 (see this module's docstring): synthesisable
    AND novel, as a conjunction. No term reads a model prediction.

    Previously a four-way conjunction (Tg window AND synthesisable AND novel
    AND ensemble-backed). Both the ``in_window`` clause and the
    ``ensemble_backed`` clause are gone, not merely dropped from the value:
    ``predictions`` is no longer read at all (see
    ``tests/test_rewards.py``'s predictor-invariance tests) --
    ``ensemble_backed`` existed only to stop a single ensemble member's guess
    from being read as a Tg consensus, and with no Tg clause there is nothing
    left for it to guard. **This arm has never been trained under this
    definition; no data exists for it.**

    This is deliberately NOT the same reward as :class:`SynthesisabilityArm`:
    that arm is a single verifiable axis (SA alone) plus the gate, mirroring
    ``NoveltyArm``; this one keeps its original character as the
    multi-criterion conjunction -- every non-Tg clause the old C4 already had,
    now the whole story instead of three quarters of it. The name is kept
    (not renamed to ``synthesisability``) for exactly that reason: it is a
    different, stricter reward than the SA-only arm, not a duplicate of it.
    Because the name is unchanged, ``configs/rl/constraint.yaml`` is
    annotated at the point every dropped key stops mattering, so a reader
    cannot mistake the old config for still producing the old reward -- see
    that file and :func:`~polyt5.rewards.constraints.constraint_reward`.
    """

    def __call__(self, candidates, targets, predictions):
        out: list[RewardResult] = []
        for pselfies, _target, _prediction in zip(candidates, targets, predictions,
                                                  strict=True):
            gate, canon = self._prepare(pselfies)
            if gate.gated:
                out.append(gate)
                continue
            sa = synthetic_accessibility(canon) if canon else None
            novel = bool(novelty_reward(canon, self.novelty_index).value)
            out.append(constraint_reward(sa, novel, sa_max=self.sa_max))
        return out


#: Third element of :class:`ControlArm`'s RNG seed tuple.
#:
#: :meth:`~polyt5.rl.trainer.GRPOTrainer.step` seeds
#: ``np.random.default_rng([cfg.seed, step_index])`` for its OWN target-Tg
#: sampling (see that module's docstring), then immediately draws
#: ``rng.uniform(target_min, target_max, size=prompts_per_step)`` from it.
#: Seeding :class:`ControlArm` with that IDENTICAL two-integer pair -- as an
#: earlier version of this class did -- means the control's first
#: ``prompts_per_step`` rewards ARE that step's normalized target values:
#: measured correlation ``1.000000``, max absolute difference ``0.0``,
#: against ``target_min + (target_max - target_min) * control_draws``. That
#: is a deterministic affine image of the run's conditioning variable, not
#: independent noise -- precisely the property this arm exists to rule out
#: (see ``ControlArm``'s own docstring).
#:
#: A THIRD seed element already guarantees a different entropy pool than the
#: trainer's two-element one, for any value -- ``numpy``'s ``SeedSequence``
#: does not collide sequences of different length. The specific value below
#: is therefore not load-bearing on its own; a named module-level constant is
#: used anyway, rather than a bare literal inlined at the call site, so the
#: intent ("this integer exists ONLY to keep this stream distinct from
#: anything else in the process, on purpose") is legible at the point of use
#: instead of resting on an implementation detail of ``SeedSequence`` hashing
#: that nothing here documents.
_CONTROL_STREAM_TAG: int = 0xC0117201


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
    * **Reproducible, not merely random - and on its OWN stream.**
      ``np.random.default_rng([seed, step_index, _CONTROL_STREAM_TAG])`` -
      deliberately built from the same two leading integers
      :meth:`~polyt5.rl.trainer.GRPOTrainer.step` uses for its own target-Tg
      sampling (see that module's docstring), but tagged with a third,
      distinct constant so it is never the SAME stream. Without the tag this
      arm and the trainer's own ``rng = np.random.default_rng([cfg.seed,
      step_index])`` would draw from the identical bit stream - and since
      the trainer's very next call is ``rng.uniform(target_min, target_max,
      size=prompts_per_step)``, the control's "independent noise" would be a
      deterministic affine image of that step's target Tg vector (measured,
      before :data:`_CONTROL_STREAM_TAG` existed: correlation ``1.000000``
      between the first ``prompts_per_step`` control rewards and that step's
      targets). That is not independent of the run's conditioning variable,
      which is exactly the property this arm exists to guarantee - see
      :data:`_CONTROL_STREAM_TAG`. Replaying step ``step_index`` of a run
      with the same ``seed`` still reproduces the exact same rewards; only
      the stream itself is guaranteed distinct from anything else in the
      process. The GLOBAL numpy RNG (``np.random.seed``/``np.random.rand``)
      is never touched either, so these draws cannot be perturbed by, or
      accidentally perturb, anything else that happens to use the global
      generator.
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
        # The third element is load-bearing, not decoration -- see
        # _CONTROL_STREAM_TAG's own docstring. [self.seed, step_index] ALONE
        # is the identical pair GRPOTrainer.step seeds its own target-Tg
        # sampling from.
        rng = np.random.default_rng([self.seed, int(step_index), _CONTROL_STREAM_TAG])
        draws = rng.uniform(0.0, 1.0, size=len(candidates))
        return [RewardResult(float(value), {"control": float(value)}) for value in draws]


#: ``accuracy`` is RETIRED (see :class:`AccuracyArm`'s docstring) but stays
#: registered so its completed run remains reproducible and reportable.
#: ``novelty`` and ``synthesisability`` are new (2026-08-23); ``composite``
#: and ``constraint`` are registered under their existing names but with
#: REDEFINED, Tg-free rewards -- see this module's docstring.
_ARMS = {"accuracy": AccuracyArm, "validity": ValidityArm,
         "novelty": NoveltyArm, "synthesisability": SynthesisabilityArm,
         "composite": CompositeArm, "constraint": ConstraintArm,
         "control": ControlArm}


def build_arm(name: str, **kwargs: Any) -> ArmReward:
    """Construct an arm by name.

    Args:
        name: One of ``accuracy`` (retired), ``validity``, ``novelty``,
            ``synthesisability``, ``composite``, ``constraint``, ``control``.
        **kwargs: Passed to the arm - ``novelty_index``, ``tolerance``,
            ``sa_max``, ``tg_config``, ``ensemble_size``, and for composite,
            ``weights``. ``accuracy`` (the only arm still reading the Tg term)
            MUST be given the ``ensemble_size`` of the predictor whose triples
            it will receive; see :class:`_BaseArm`. ``control`` additionally
            reads ``seed`` (default ``0``) - see :class:`ControlArm`.

    Raises:
        ValueError: On an unknown arm name.
    """
    if name not in _ARMS:
        raise ValueError(f"unknown arm {name!r}; expected one of {sorted(_ARMS)}")
    return _ARMS[name](**kwargs)
