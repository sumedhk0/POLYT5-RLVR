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
from polyt5.rl.grpo import GRPOConfig, grpo_loss
from polyt5.rl.reference_policy import ReferencePolicy
from polyt5.rl.rollout import ROLLOUT_CHUNK_SIZE, sample_groups
from polyt5.tokenization import PolyT5Tokenizer
from polyt5.training import build_optimizer, save_checkpoint
from polyt5.utils import RunDirectory
from polyt5.utils.device import select_device

__all__ = ["GRPOTrainer", "GRPOTrainerConfig"]


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

    def __post_init__(self) -> None:
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
        # ReferencePolicy mutates `reference` in place (.eval(),
        # requires_grad_(False)) rather than copying it -- see its docstring.
        # `reference` must already be a model instance distinct from `policy`;
        # this class never constructs one from the live policy object.
        self.reference = ReferencePolicy(reference, device=self.device)
        self.optimizer = build_optimizer(
            self.policy, lr=config.learning_rate, weight_decay=config.weight_decay
        )

    def step(self, step_index: int) -> dict[str, float]:
        """Run one GRPO step and return its logging stats.

        Args:
            step_index: 0-based step index, mixed into the step's RNG seed so
                the step is reproducible independent of history (see the
                module docstring).

        Returns:
            Stats dict with (at least) ``reward_mean``, ``reward_unweighted_mean``,
            ``kl``, ``clip_fraction``, ``gated_fraction``, ``mean_length`` and
            ``loss``.
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
        results: list[RewardResult] = self.arm(batch.texts, batch.targets, predictions)

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

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        unweighted = [r.components.get("closeness", 0.0) for r in results]
        gated = [float(r.gated) for r in results]
        mean_length = float(batch.mask.sum(dim=1).float().mean().item())

        return {
            "reward_mean": float(rewards.mean()),
            "reward_unweighted_mean": float(np.mean(unweighted)),
            "kl": loss_stats["kl"],
            "clip_fraction": loss_stats["clip_fraction"],
            "mean_ratio": loss_stats["mean_ratio"],
            "policy_loss": loss_stats["policy_loss"],
            "gated_fraction": float(np.mean(gated)),
            "mean_length": mean_length,
            "loss": float(loss.detach()),
        }

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

    def train(self) -> dict[str, Any]:
        """Run ``config.max_steps`` steps, logging and checkpointing when a run dir is given.

        Logging and checkpointing cadence are both keyed off the number of
        COMPLETED steps (``step_index + 1``), so ``log_every`` and
        ``save_every`` mean the same thing, and both ALWAYS fire on the final
        step regardless of whether ``max_steps`` is a multiple of the cadence
        -- a run that stops mid-cadence still logs and checkpoints its last
        weights rather than silently dropping them.

        Returns:
            ``{"num_steps": int, "history": list[dict]}`` -- every step's stats,
            in order.
        """
        history: list[dict[str, float]] = []
        for step_index in range(self.config.max_steps):
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

        return {"num_steps": self.config.max_steps, "history": history}
