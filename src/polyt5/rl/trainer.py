"""The synchronous GRPO trainer: wires sampling, reward, advantage and loss together.

Per step: sample target Tg values, sample a GROUP of candidates per target,
score them with the verifiable-chemistry arm plus a Tg ensemble, convert
rewards to group-relative advantages, recompute log-probs under the policy
WITH gradients, compute the clipped surrogate plus a KL anchor to a frozen
reference, then backward / clip / step.

Determinism: a step is reproducible from ``(config.seed, step_index)`` alone,
independent of any prior step or call history. A single
``np.random.default_rng([seed, step_index])`` supplies both the
uniformly-sampled target Tg values AND -- via a further draw from that SAME
generator -- the rollout's sampling seed, so re-running one step twice (even
on a freshly constructed trainer) samples the same targets and the same
candidates.

Dropout and module mode: :func:`~polyt5.rl.rollout.sample_groups` and
:meth:`~polyt5.rl.reference_policy.ReferencePolicy.score` both document that
they do not change the model's train/eval mode themselves -- callers decide.
This trainer keeps the policy in ``eval()`` for BOTH the rollout sampling
AND the log-prob recompute that feeds the loss (the reference policy is
always ``eval()`` for its whole lifetime regardless; see
:class:`~polyt5.rl.reference_policy.ReferencePolicy`). There is exactly one
gradient step per rollout, so ``pi_theta`` and ``pi_theta_old`` are evaluated
at IDENTICAL parameters; running the recompute in ``train()`` mode would make
them differ only by dropout noise between two passes over the same weights,
which is not policy drift -- it corrupts the importance ratio (spurious
clipping at step 0, before any update has happened) and the KL anchor (a
permanent positive floor from comparing a dropped-out pass against the
reference's always-clean one, fighting the KL term's own purpose). Keeping
both passes in ``eval()`` makes the ratio exactly ``1.0`` at step 0 and the
surrogate reduce to the vanilla policy-gradient update, which is correct here
precisely because training is on-policy for exactly one step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

from polyt5.rewards import ArmReward, RewardResult
from polyt5.rl.advantages import group_advantages
from polyt5.rl.drift import DriftMonitor
from polyt5.rl.grpo import GRPOConfig, grpo_loss
from polyt5.rl.reference_policy import ReferencePolicy
from polyt5.rl.rollout import ROLLOUT_CHUNK_SIZE, sample_groups
from polyt5.tokenization import PolyT5Tokenizer
from polyt5.training import build_optimizer, save_checkpoint
from polyt5.utils import RunDirectory, get_logger
from polyt5.utils.device import select_device

__all__ = ["GRPOTrainer", "GRPOTrainerConfig"]

_LOGGER = get_logger("polyt5.rl.trainer")

#: Per-arm reward component keys worth aggregating into the step stats. Every
#: one is a fraction or a mean in natural units, and every one is ``None`` in
#: the stats dict for an arm that does not produce it -- never a plausible
#: zero. Keys the arms actually emit:
#:   accuracy   closeness confidence abs_error std sigma_effective
#:              n_contributing coverage
#:   validity   sv tsd dd pv
#:   composite  the accuracy set + novelty + pv
#:   constraint in_window synthesisable novel ensemble_backed coverage
#:              n_contributing
_DIAGNOSTIC_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("sv_pass_rate", "sv"),
    ("tsd_pass_rate", "tsd"),
    ("dd_pass_rate", "dd"),
    ("pv_pass_rate", "pv"),
    ("novelty_mean", "novelty"),
    ("novel_rate", "novel"),
    ("in_window_rate", "in_window"),
    ("synthesisable_rate", "synthesisable"),
    ("ensemble_backed_rate", "ensemble_backed"),
    ("closeness_mean", "closeness"),
    ("confidence_mean", "confidence"),
    ("abs_error_mean", "abs_error"),
    ("ensemble_std_mean", "std"),
)

#: Fraction of groups with zero reward variance above which the step is
#: reported as a probable no-op. Every candidate in such a group gets an
#: advantage of exactly 0.0 (see :func:`~polyt5.rl.advantages.group_advantages`),
#: so those groups contribute no policy gradient at all -- only the KL anchor
#: survives, while ``reward_mean`` still reads a plausible small number.
_DEGENERATE_GROUP_FRACTION = 0.9


def _component_mean(results: Sequence[RewardResult], key: str) -> float | None:
    """Mean of one reward component across the batch, or ``None`` if absent.

    Returns ``None`` -- never ``0.0`` -- when NO result carries ``key``: an
    arm that does not produce a component has not measured it at zero, and a
    metric that reads ``0.0`` for all 2000 steps of a run is indistinguishable
    from a bug.
    """
    values = [result.components[key] for result in results if key in result.components]
    if not values:
        return None
    return float(sum(values) / len(values))


class UncertaintyPredictor(Protocol):
    """Duck-typed contract for the Tg ensemble predictor.

    Matches :class:`polyt5.inference.predictor.PolyT5PropertyEnsemble` (and
    the brief's ``_FakePredictor`` test double) without requiring either to
    inherit from this class.
    """

    def predict_with_uncertainty(
        self, candidates: Sequence[str]
    ) -> Sequence[tuple[float, float, int]]:
        """Return one ``(mean, std, n_contributing_members)`` triple per candidate."""
        ...


#: Third element of the replay batch's RNG seed. A distinct stream tag keeps the
#: replay draw from mirroring the rollout draw, which shares (seed, step_index) --
#: the control arm learned that lesson when its reward correlated 1.000000 with the
#: target because both used the same two-element seed.
REPLAY_STREAM = 20260831


@dataclass(frozen=True)
class GRPOTrainerConfig:
    """Hyperparameters for :class:`GRPOTrainer`.

    Attributes:
        group_size: Candidates sampled per target Tg value.
        prompts_per_step: Distinct target Tg values sampled per step.
        max_steps: Number of steps :meth:`GRPOTrainer.train` runs.
        max_length: Maximum generated tokens per candidate.
        target_min: Lower bound of the uniform target-sampling range, Kelvin.
        target_max: Upper bound of the uniform target-sampling range, Kelvin.
        temperature: Sampling temperature for rollout generation.
        top_p: Nucleus mass for rollout generation.
        learning_rate: AdamW learning rate for the policy.
        weight_decay: Decoupled weight decay, passed to
            :func:`~polyt5.training.optim.build_optimizer` (which excludes
            embeddings, biases and LayerNorm/T5LayerNorm scales from decay --
            see its docstring). Defaults to ``0.0``: for a policy-gradient
            recompute, zero decay means a parameter moves only when it
            actually received a non-zero gradient, which is what makes
            "did training happen" testable by inspecting gradients rather
            than by asserting on decay-induced parameter drift.
        clip_eps: GRPO ratio clipping half-width (see
            :class:`~polyt5.rl.grpo.GRPOConfig`).
        kl_coef: Weight on the KL anchor to the frozen reference policy.
        max_grad_norm: Global-norm gradient clipping threshold.
        device: ``"auto"``, ``"cuda"``, or ``"cpu"`` (resolved via
            :func:`~polyt5.utils.device.select_device`).
        seed: Base seed. Combined with a step index to seed a private
            ``np.random.default_rng`` for that step alone (see the module
            docstring).
        log_every: Log metrics every this many steps in :meth:`GRPOTrainer.train`.
        save_every: Checkpoint every this many steps in :meth:`GRPOTrainer.train`
            (the final step always checkpoints too, regardless of this value).
        rollout_batch_size: Candidates per :func:`~polyt5.generation.generate`
            call during rollout, forwarded to :func:`~polyt5.rl.rollout.
            sample_groups` as its ``chunk_size`` argument. Defaults to
            :data:`polyt5.rl.rollout.ROLLOUT_CHUNK_SIZE` (128), the measured
            hardware optimum documented on that module -- lowering this trades
            throughput (measured ~4x worse at 512) for peak memory; it does
            not change what gets generated, only how it is batched.
        replay_coef: Weight on the supervised replay term (round 3, see
            ``docs/superpowers/specs/2026-08-31-supervised-replay-design.md``).
            ``0.0`` -- the default -- makes the step numerically identical to a
            trainer without replay, which is what makes the feature safe to add
            mid-study. Non-zero REQUIRES ``replay_dataset``; the trainer raises
            rather than training without it, because a run that believes it is
            doing replay and is not would read as evidence that replay fails.
        replay_batch_size: Supervised pairs drawn per step when replay is on.
        drift_every: Run the spec section 4.4 drift monitor (see
            :class:`~polyt5.rl.drift.DriftMonitor`) on step indices divisible
            by this, when a monitor is attached. Step 0 is always measured, so
            every run records a baseline before any update. Costs one extra
            auditor pass plus a fingerprint sweep on the measured steps only;
            at the default 50 that is ~2% of a run.
    """

    group_size: int = 16
    prompts_per_step: int = 32
    max_steps: int = 2000
    max_length: int = 200
    target_min: float = 250.0
    target_max: float = 600.0
    temperature: float = 0.7
    top_p: float = 0.95
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    clip_eps: float = 0.2
    kl_coef: float = 0.02
    max_grad_norm: float = 1.0
    device: str = "auto"
    seed: int = 0
    log_every: int = 10
    save_every: int = 250
    rollout_batch_size: int = ROLLOUT_CHUNK_SIZE
    drift_every: int = 50
    replay_coef: float = 0.0
    replay_batch_size: int = 16

    def __post_init__(self) -> None:
        if self.drift_every < 1:
            raise ValueError(f"drift_every must be >= 1, got {self.drift_every}")
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")
        if self.prompts_per_step < 1:
            raise ValueError(f"prompts_per_step must be >= 1, got {self.prompts_per_step}")
        if self.max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {self.max_length}")
        if self.target_min >= self.target_max:
            raise ValueError(
                f"target_min ({self.target_min}) must be < target_max ({self.target_max})"
            )
        if self.rollout_batch_size < 1:
            raise ValueError(f"rollout_batch_size must be >= 1, got {self.rollout_batch_size}")


class GRPOTrainer:
    """Synchronous GRPO trainer: one policy, one frozen reference, one loop.

    See the module docstring for the per-step algorithm and the determinism
    and dropout-handling rationale.
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        reference: torch.nn.Module,
        tokenizer: PolyT5Tokenizer,
        arm: ArmReward,
        predictor: UncertaintyPredictor,
        config: GRPOTrainerConfig,
        run_dir: RunDirectory | None = None,
        drift_monitor: DriftMonitor | None = None,
        replay_dataset: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        """Build the trainer.

        Args:
            policy: The model being trained. Moved to ``config.device`` in
                place; its parameters receive gradients.
            reference: A SEPARATE model instance (never the same object as
                ``policy``) used to build the frozen reference policy. Frozen
                and moved to ``config.device`` in place -- see
                :class:`~polyt5.rl.reference_policy.ReferencePolicy` for why
                handing it ``policy`` itself would silently freeze the policy.
            tokenizer: Tokenizer shared by rollout sampling and checkpoint
                provenance (``tokenizer.sha256``).
            arm: Reward arm, e.g. ``build_arm("accuracy")``.
            predictor: Tg ensemble predictor exposing
                ``predict_with_uncertainty``.
            config: Trainer hyperparameters.
            run_dir: Optional :class:`~polyt5.utils.RunDirectory` for metric
                logging and checkpointing in :meth:`train`. ``step`` alone
                never touches it.
            drift_monitor: Optional :class:`~polyt5.rl.drift.DriftMonitor`
                implementing spec section 4.4. It is held in its OWN attribute
                and is never passed to ``arm``, never consulted before or
                during reward computation, and its output only ever reaches the
                returned stats dict -- so a monitor carrying the held-out
                auditor cannot reach a reward path. ``tests/test_rl_drift.py``
                pins that the step's rewards, advantages and loss are identical
                with and without it.

        Raises:
            ValueError: If ``reference is policy`` -- see the ``reference``
                argument above. Without this guard the failure is not silent
                (``ReferencePolicy`` clearing ``requires_grad`` on the shared
                object makes ``loss.backward()`` raise), but that indirect
                failure mode is exactly the shape of bug that could otherwise
                burn hours of a real run before surfacing, so it gets a named
                error instead.
        """
        if reference is policy:
            raise ValueError(
                "GRPOTrainer.reference must be a SEPARATE model instance from `policy`: "
                "ReferencePolicy freezes the object it is given in place "
                "(.eval(), requires_grad_(False)), so handing it `policy` itself would "
                "silently freeze the policy too. Load a separate checkpoint or construct a "
                "second model instance instead."
            )
        self.config = config
        self.device = torch.device(select_device(config.device))
        self.policy = policy.to(self.device)
        self.tokenizer = tokenizer
        self.arm = arm
        self.predictor = predictor
        self.run_dir = run_dir
        # Held separately from `self.arm` on purpose: the monitor may carry the
        # held-out auditor, which must never reach a reward path. Nothing in
        # this class passes one to the other.
        self.drift_monitor = drift_monitor
        # A non-zero coefficient with no data must FAIL, never silently train
        # without replay: such a run would look like evidence that replay does
        # not work, which is the most expensive way for this to go wrong.
        if config.replay_coef and not replay_dataset:
            raise ValueError(
                f"replay_coef={config.replay_coef} requires replay_dataset; got none. "
                "Training without the supervised term would silently produce a run that "
                "looks like replay and is not."
            )
        if config.replay_coef < 0:
            raise ValueError(f"replay_coef must be non-negative, got {config.replay_coef}")
        self.replay_dataset = list(replay_dataset) if replay_dataset else []
        # ReferencePolicy mutates `reference` in place (.eval(),
        # requires_grad_(False)) rather than copying it -- see its docstring.
        # `reference` must already be a model instance distinct from `policy`;
        # this class never constructs one from the live policy object.
        self.reference = ReferencePolicy(reference, device=self.device, tokenizer=tokenizer)
        self.optimizer = build_optimizer(
            self.policy, lr=config.learning_rate, weight_decay=config.weight_decay
        )

    def step(self, step_index: int) -> dict[str, Any]:
        """Run one GRPO step and return its logging stats.

        Args:
            step_index: 0-based step index, mixed into the step's RNG seed so
                the step is reproducible independent of history (see the
                module docstring).

        Returns:
            Stats dict with (at least) ``reward_mean``, ``reward_unweighted_mean``,
            ``kl``, ``clip_fraction``, ``gated_fraction``, ``mean_length`` and
            ``loss``.

            Every arm's degenerate optimum used to be invisible until
            ``scripts/compare_arms.py`` ran, roughly seven hours after the run
            started, because the only reward-side numbers logged were
            ``reward_mean`` and a ``closeness`` mean that two of the four arms
            never produce. The stats therefore also carry, per step:

            * the **partial-ensemble** counters
              (``ensemble_full_fraction`` / ``ensemble_partial_fraction`` /
              ``ensemble_empty_fraction`` / ``mean_contributing_members``) --
              the direct in-flight detector for a policy walking off the
              reward ensemble's support;
            * the **cascade and novelty** rates the arm actually measured
              (``sv_pass_rate``, ``tsd_pass_rate``, ``dd_pass_rate``,
              ``pv_pass_rate``, ``novelty_mean``, ``novel_rate``, ...), each
              ``None`` for an arm that does not produce it;
            * the **collapse** counters ``unique_fraction``,
              ``zero_variance_group_fraction`` and
              ``nonzero_advantage_fraction``, which together detect the
              zero-gradient no-op that mode collapse produces under C2's
              cross-group DD dedup while ``reward_mean`` still reads a
              plausible small number;
            * spec section 4.4's ``drift_*`` keys on monitored steps, when a
              :class:`~polyt5.rl.drift.DriftMonitor` is attached.
        """
        cfg = self.config
        rng = np.random.default_rng([cfg.seed, step_index])
        targets = rng.uniform(cfg.target_min, cfg.target_max, size=cfg.prompts_per_step).tolist()
        # A further draw from the SAME step-seeded generator, so the whole
        # step -- not just the target values -- is pinned by (seed, step_index).
        rollout_seed = int(rng.integers(0, 2**31 - 1))

        self.policy.eval()
        batch = sample_groups(
            self.policy,
            self.tokenizer,
            targets=targets,
            group_size=cfg.group_size,
            max_length=cfg.max_length,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            seed=rollout_seed,
            device=self.device,
            chunk_size=cfg.rollout_batch_size,
        )

        predictions = self.predictor.predict_with_uncertainty(batch.texts)
        # Step-aware arms (currently only ControlArm -- see its docstring)
        # declare `wants_step_index = True` and read an extra keyword-only
        # `step_index` beyond the shared three-argument ArmReward contract,
        # so their own reward can be seeded from (config.seed, step_index)
        # the same way this step already seeds `targets` above. Every other
        # arm's __call__ does not accept that keyword at all, so it must
        # never be passed unconditionally.
        if getattr(self.arm, "wants_step_index", False):
            results: list[RewardResult] = self.arm(
                batch.texts, batch.targets, predictions, step_index=step_index
            )
        else:
            results = self.arm(batch.texts, batch.targets, predictions)

        rewards = np.array([r.value for r in results], dtype=np.float64)
        advantages = group_advantages(rewards, cfg.group_size)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        # Still eval() -- see module docstring for why the recompute must NOT
        # switch to train() mode: exactly one gradient step per rollout means
        # pi_theta and pi_theta_old are the same distribution at recompute
        # time, and only eval() keeps that true (dropout would inject noise
        # unrelated to any actual policy update). Gradients still flow in
        # eval mode -- eval()/train() governs dropout and batchnorm, not
        # autograd.
        logprobs = self._policy_logprobs(batch.sequences, batch.prompt_ids, batch.prompt_mask)
        # ReferencePolicy.score() returns UNMASKED log-probs; grpo_loss applies
        # `mask` itself, so no separate masking step is needed here.
        ref_logprobs = self.reference.score(batch.sequences, batch.prompt_ids, batch.prompt_mask)

        grpo_config = GRPOConfig(clip_eps=cfg.clip_eps, kl_coef=cfg.kl_coef)
        loss, loss_stats = grpo_loss(
            logprobs,
            batch.logprobs,
            ref_logprobs,
            advantages_t,
            batch.mask,
            config=grpo_config,
        )

        # Added BEFORE the single backward pass on purpose: a second
        # optimizer.step() would double the effective learning rate on replay
        # batches and desynchronise the step budget from every other arm.
        replay = self._replay_loss(step_index)
        if replay is not None:
            loss = loss + cfg.replay_coef * replay

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        gated = [float(r.gated) for r in results]
        mean_length = float(batch.mask.sum(dim=1).float().mean().item())

        stats: dict[str, Any] = {
            # Logged unconditionally. Without it a run where replay silently
            # contributes nothing is indistinguishable from one where it works.
            "replay_coef": cfg.replay_coef,
            "replay_loss": (float(replay.item()) if replay is not None else None),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            # `closeness` is the confidence weight's OWN unweighted counterpart
            # and only the two Tg-scoring arms produce it. For `validity` and
            # `constraint` this is None, not 0.0: reading `0.0` for all 2000
            # steps of a run is indistinguishable from a broken metric, and
            # spec 4.4's claim ("the gate's effect on the learning signal is
            # measurable rather than assumed") is simply not true for an arm
            # that has no gate to measure.
            "reward_unweighted_mean": _component_mean(results, "closeness"),
            "kl": loss_stats["kl"],
            "clip_fraction": loss_stats["clip_fraction"],
            "mean_ratio": loss_stats["mean_ratio"],
            "policy_loss": loss_stats["policy_loss"],
            "gated_fraction": float(np.mean(gated)),
            "mean_length": mean_length,
            "loss": float(loss.detach()),
            **self._diagnostics(batch.texts, predictions, results, rewards, advantages),
        }

        if self.drift_monitor is not None and step_index % cfg.drift_every == 0:
            drift = self.drift_monitor.observe(batch.texts, predictions)
            stats.update({f"drift_{key}": value for key, value in drift.items()})

        self._warn_on_degenerate_step(step_index, stats)
        return stats

    def _diagnostics(
        self,
        texts: Sequence[str],
        predictions: Sequence[tuple[float, float, int]],
        results: Sequence[RewardResult],
        rewards: np.ndarray,
        advantages: np.ndarray,
    ) -> dict[str, Any]:
        """Per-step reward-side diagnostics, computed from data already in hand.

        Args:
            texts: The rollout's generated strings.
            predictions: The property ensemble's ``(mean, std, n)`` triples.
            results: The arm's per-candidate results.
            rewards: The scalar rewards fed to :func:`group_advantages`.
            advantages: The group-relative advantages fed to the loss.

        Returns:
            A stats fragment. Nothing here costs an extra model call.
        """
        n_total = len(self.predictor) if hasattr(self.predictor, "__len__") else None
        contributing = [int(prediction[2]) for prediction in predictions]
        n_candidates = len(contributing) or 1

        out: dict[str, Any] = {
            "unique_fraction": (len(set(texts)) / len(texts)) if texts else None,
            "mean_contributing_members": float(np.mean(contributing)) if contributing else None,
            "ensemble_empty_fraction": (
                sum(1 for n in contributing if n == 0) / n_candidates if contributing else None
            ),
        }
        if n_total is not None and contributing:
            out["ensemble_size"] = float(n_total)
            out["ensemble_full_fraction"] = (
                sum(1 for n in contributing if n >= n_total) / n_candidates
            )
            out["ensemble_partial_fraction"] = (
                sum(1 for n in contributing if 0 < n < n_total) / n_candidates
            )
        else:
            out["ensemble_size"] = None
            out["ensemble_full_fraction"] = None
            out["ensemble_partial_fraction"] = None

        for stat_key, component_key in _DIAGNOSTIC_COMPONENTS:
            out[stat_key] = _component_mean(results, component_key)

        # Zero-variance groups get an advantage of exactly 0.0 for every
        # member, so they contribute NO policy gradient. Under mode collapse
        # C2 produces a whole step of them while reward_mean still reads a
        # small positive number -- these two counters are what make that
        # visible at step 50 instead of at hour 7.
        group_size = self.config.group_size
        if group_size > 0 and rewards.size % group_size == 0 and rewards.size:
            grouped = rewards.reshape(-1, group_size)
            out["zero_variance_group_fraction"] = float(
                np.mean(grouped.std(axis=1) <= 1e-12)
            )
        else:
            out["zero_variance_group_fraction"] = None
        out["nonzero_advantage_fraction"] = (
            float(np.mean(np.abs(advantages) > 1e-8)) if advantages.size else None
        )
        return out

    def _warn_on_degenerate_step(self, step_index: int, stats: dict[str, Any]) -> None:
        """Log LOUDLY when a step could not have taught the policy anything.

        Two failure modes read as healthy numbers in ``reward_mean`` alone and
        are otherwise invisible until the comparison matrix runs hours later:

        * every group has zero reward variance, so every advantage is exactly
          zero and the only surviving gradient is the KL anchor pulling back
          toward the reference (the mode-collapse / cross-group-DD no-op);
        * most candidates are scored by only part of the property ensemble,
          which is the signature of a policy leaving the predictors' support.
        """
        zero_variance = stats.get("zero_variance_group_fraction")
        nonzero_advantage = stats.get("nonzero_advantage_fraction")
        if nonzero_advantage == 0.0:
            _LOGGER.warning(
                "step %d: EVERY advantage is zero -- this step's policy gradient is exactly "
                "zero and only the KL anchor moved the policy. reward_mean=%.6f "
                "unique_fraction=%s. Under mode collapse ValidityArm's cross-group DD dedup "
                "produces exactly this while reward_mean still reads a plausible small "
                "number.", step_index, stats.get("reward_mean", float("nan")),
                stats.get("unique_fraction"),
            )
        elif zero_variance is not None and zero_variance >= _DEGENERATE_GROUP_FRACTION:
            _LOGGER.warning(
                "step %d: %.0f%% of groups have zero reward variance and therefore contribute "
                "no policy gradient (reward_mean=%.6f, unique_fraction=%s).",
                step_index, 100.0 * zero_variance, stats.get("reward_mean", float("nan")),
                stats.get("unique_fraction"),
            )

        partial = stats.get("ensemble_partial_fraction")
        empty = stats.get("ensemble_empty_fraction")
        if partial is not None and empty is not None and (partial + empty) >= 0.5:
            _LOGGER.warning(
                "step %d: %.0f%% of candidates were scored by only PART of the property "
                "ensemble (partial=%.3f empty=%.3f). That is the signature of a policy "
                "leaving the reward models' support; the confidence weight is discounting "
                "these, but the trend is what matters.",
                step_index, 100.0 * (partial + empty), partial, empty,
            )

    def _replay_loss(self, step_index: int) -> Tensor | None:
        """Teacher-forced cross-entropy on a batch of the ORIGINAL supervised pairs.

        The conditioning skill lives entirely in those 6,619 ``(target, polymer)``
        examples, and GRPO's objective never mentions the target -- so without this
        term the skill is simply overwritten. See the round-3 spec.

        The batch is drawn from a generator seeded by ``(config.seed, step_index)``,
        the same rule the rollout uses, so a RESUMED step replays exactly the batch an
        uninterrupted run would have drawn.

        Args:
            step_index: 0-based step index, mixed into the batch-sampling seed.

        Returns:
            Scalar mean token cross-entropy, or ``None`` when replay is disabled.
        """
        cfg = self.config
        if not cfg.replay_coef or not self.replay_dataset:
            return None
        rng = np.random.default_rng([cfg.seed, step_index, REPLAY_STREAM])
        n = min(cfg.replay_batch_size, len(self.replay_dataset))
        picks = rng.choice(len(self.replay_dataset), size=n, replace=False)
        sources = [self.replay_dataset[int(i)][0] for i in picks]
        targets = [self.replay_dataset[int(i)][1] for i in picks]

        src = self.tokenizer.batch_encode(
            sources, add_eos=True, max_length=cfg.max_length, padding=True, truncation=True
        )
        tgt = self.tokenizer.batch_encode(
            targets, add_eos=True, max_length=cfg.max_length, padding=True, truncation=True
        )
        input_ids = torch.tensor(src["input_ids"], device=self.device)
        attention_mask = torch.tensor(src["attention_mask"], device=self.device)
        labels = torch.tensor(tgt["input_ids"], device=self.device)
        label_mask = torch.tensor(tgt["attention_mask"], device=self.device).bool()

        start = torch.full(
            (labels.size(0), 1), self.tokenizer.decoder_start_token_id, device=self.device
        )
        decoder_input_ids = torch.cat([start, labels[:, :-1]], dim=1)
        # eval() explicitly, not inherited from the caller. Dropout would make this
        # term nondeterministic, so a resumed step would compute a different replay
        # loss from the same batch and resume would stop being bit-identical. Setting
        # it here rather than relying on step() having done so also keeps the helper
        # correct when called directly. eval() governs dropout, not autograd -- the
        # gradient still flows.
        was_training = self.policy.training
        self.policy.eval()
        try:
            logits = self.policy(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            ).logits
        finally:
            self.policy.train(was_training)
        token_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none"
        ).view(labels.shape)
        # Mean over REAL tokens only; padding would otherwise dilute the loss by a
        # factor that varies with batch composition.
        return (token_loss * label_mask).sum() / label_mask.sum().clamp(min=1)

    def _policy_logprobs(
        self, sequences: Tensor, prompt_ids: Tensor, prompt_mask: Tensor
    ) -> Tensor:
        """Teacher-forced per-token log pi_theta of ``sequences``, WITH gradient.

        Mirrors :meth:`~polyt5.rl.reference_policy.ReferencePolicy.score`
        exactly, minus its ``@torch.no_grad()`` -- this is the one place in
        the step that must build a graph back to the policy's parameters.

        Args:
            sequences: ``(n, gen_len)`` decoder-side token ids (decoder start
                token NOT included).
            prompt_ids: ``(n, src_len)`` encoder input ids.
            prompt_mask: ``(n, src_len)`` encoder padding mask.

        Returns:
            ``(n, gen_len)`` log-probs of every position in ``sequences``
            under the CURRENT policy, unmasked (mirrors
            :meth:`ReferencePolicy.score`'s own contract; :func:`~polyt5.rl.
            grpo.grpo_loss` applies the mask).

        Raises:
            ValueError: If the tokenizer's ``decoder_start_token_id`` (the id
                :func:`~polyt5.rl.rollout.sample_groups` actually shifted in
                when it built ``sequences``) disagrees with the policy's own
                ``config.decoder_start_token_id``. A silent mismatch here
                would desynchronise every position of ``logprobs`` from
                ``old_logprobs`` by one token without raising anywhere.
        """
        if self.tokenizer.decoder_start_token_id != self.policy.config.decoder_start_token_id:
            raise ValueError(
                "tokenizer.decoder_start_token_id "
                f"({self.tokenizer.decoder_start_token_id}) != "
                f"policy.config.decoder_start_token_id "
                f"({self.policy.config.decoder_start_token_id}): sample_groups() shifted in "
                "the tokenizer's start token when it built `sequences`, so recomputing "
                "log-probs with a different one would silently desynchronise every position."
            )
        sequences = sequences.to(self.device)
        prompt_ids = prompt_ids.to(self.device)
        prompt_mask = prompt_mask.to(self.device)

        decoder_start = self.tokenizer.decoder_start_token_id
        start = torch.full(
            (sequences.shape[0], 1), decoder_start, dtype=sequences.dtype, device=self.device
        )
        decoder_input_ids = torch.cat([start, sequences[:, :-1]], dim=1)
        output = self.policy(
            prompt_ids, attention_mask=prompt_mask, decoder_input_ids=decoder_input_ids
        )
        logprobs = torch.log_softmax(output.logits.float(), dim=-1)
        return logprobs.gather(2, sequences.unsqueeze(-1)).squeeze(-1)

    def train(self, start_step: int = 0) -> dict[str, Any]:
        """Run steps ``start_step .. config.max_steps - 1``, logging/checkpointing along the way.

        Logging and checkpointing cadence are both keyed off the number of
        COMPLETED steps (``step_index + 1``), so ``log_every`` and
        ``save_every`` mean the same thing, and both ALWAYS fire on the final
        step regardless of whether ``max_steps`` is a multiple of the cadence
        -- a run that stops mid-cadence still logs and checkpoints its last
        weights rather than silently dropping them.

        Args:
            start_step: 0-based step index to start at. ``0`` (the default)
                runs every step from the beginning, exactly as before this
                parameter existed. A caller resuming from a checkpoint whose
                ``global_step`` is ``N`` should pass ``start_step=N``: a
                step's RNG is seeded from ``(config.seed, step_index)`` alone
                (see the module docstring), so restarting at ``step_index 0``
                would silently replay the exact rollouts the checkpoint
                already trained on instead of continuing past them.

        Returns:
            ``{"num_steps": int, "history": list[dict]}`` -- ``num_steps`` is
            the number of steps actually run (``max_steps - start_step``,
            floored at 0), and ``history`` holds every one of those steps'
            stats, in order.
        """
        history: list[dict[str, Any]] = []
        for step_index in range(start_step, self.config.max_steps):
            stats = self.step(step_index)
            history.append(stats)

            completed = step_index + 1
            is_final = completed == self.config.max_steps
            if self.run_dir is not None:
                if completed % self.config.log_every == 0 or is_final:
                    self.run_dir.log_metrics({"step": completed, **stats})
                if completed % self.config.save_every == 0 or is_final:
                    save_checkpoint(
                        self.run_dir.checkpoints / f"step_{completed:06d}.pt",
                        model=self.policy,
                        optimizer=self.optimizer,
                        epoch=0,
                        global_step=completed,
                        config={"grpo": asdict(self.config)},
                        model_config=self.policy.config.to_dict(),
                        tokenizer_sha256=self.tokenizer.sha256,
                        train_metrics=stats,
                    )

        return {"num_steps": max(0, self.config.max_steps - start_step), "history": history}
