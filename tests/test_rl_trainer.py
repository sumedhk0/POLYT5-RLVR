# tests/test_rl_trainer.py
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.rewards import TgRewardConfig, build_arm
from polyt5.rl.trainer import GRPOTrainer, GRPOTrainerConfig
from polyt5.tokenization import PolyT5Tokenizer


class _FakePredictor:
    """Returns (mean, std, n) triples; deterministic, no torch."""

    def __init__(self, mean=500.0, std=5.0):
        self.mean, self.std = mean, std
        self.calls = 0

    def predict_with_uncertainty(self, candidates):
        self.calls += 1
        return [(self.mean, self.std, 4) for _ in candidates]


def _tiny():
    tok = PolyT5Tokenizer.default()
    cfg = PolyT5Config(vocab_size=tok.vocab_size, d_model=64, d_kv=16, num_heads=4,
                       d_ff=128, num_layers=2, n_positions=64,
                       pad_token_id=tok.pad_id, eos_token_id=tok.eos_id,
                       decoder_start_token_id=tok.decoder_start_token_id)
    return PolyT5ForConditionalGeneration(cfg), tok


def test_one_step_runs_end_to_end_and_changes_parameters(monkeypatch):
    """Also verifies real learning occurred (a non-zero gradient), not merely
    that SOMETHING moved the parameters.

    A fresh, untrained policy structurally gates virtually every candidate it
    actually samples (verified empirically: 6/6 random-weight trials come back
    100% gated), which makes the surrogate term exactly zero by
    `group_advantages`'s own documented "all-identical -> zero" behaviour. With
    both the rollout and the recompute correctly kept in `eval()` (see
    task-7-review.md finding 1) and `weight_decay=0.0` (finding 2), that would
    make the WHOLE step a genuine, gradient-free no-op -- which is CORRECT
    training semantics, not a bug to route around in trainer.py. So this test
    forces a non-degenerate reward signal instead, by monkeypatching
    `sample_groups` to alternate a real, valid polymer with structurally
    invalid garbage within each group (a wide accuracy tolerance keeps the
    valid candidate's reward away from zero regardless of which random target
    this step happened to draw), and then asserts the resulting gradient norm
    is non-zero -- not just that parameters moved, which `torch.optim.AdamW`'s
    default `weight_decay=0.01` used to satisfy on its own even with an
    exactly-zero gradient (see task-7-review.md section 4).
    """
    import polyt5.rl.trainer as trainer_mod

    policy, tok = _tiny()
    reference, _ = _tiny()
    reference.load_state_dict(policy.state_dict())
    before = [p.detach().clone() for p in policy.parameters()]

    real_sample_groups = trainer_mod.sample_groups

    def fake_sample_groups(*args, **kwargs):
        batch = real_sample_groups(*args, **kwargs)
        n = len(batch.texts)
        forced = ["[At][C][C][O][At]" if i % 2 == 0 else "not a polymer" for i in range(n)]
        return dataclasses.replace(batch, texts=forced)

    monkeypatch.setattr(trainer_mod, "sample_groups", fake_sample_groups)

    trainer = GRPOTrainer(
        policy=policy, reference=reference, tokenizer=tok,
        arm=build_arm("accuracy", tg_config=TgRewardConfig(tolerance=1000.0)),
        predictor=_FakePredictor(),
        config=GRPOTrainerConfig(group_size=4, prompts_per_step=2, max_length=32,
                                 device="cpu", seed=0, learning_rate=1e-3),
    )
    stats = trainer.step(0)
    assert np.isfinite(stats["reward_mean"])
    assert np.isfinite(stats["loss"])

    grad_norm_sq = sum(
        float(p.grad.detach().pow(2).sum())
        for p in trainer.policy.parameters()
        if p.grad is not None
    )
    assert grad_norm_sq > 0.0, "no gradient reached the policy"

    after = list(policy.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_reference_policy_never_updates():
    policy, tok = _tiny()
    reference, _ = _tiny()
    ref_before = [p.detach().clone() for p in reference.parameters()]
    trainer = GRPOTrainer(
        policy=policy, reference=reference, tokenizer=tok,
        arm=build_arm("accuracy"), predictor=_FakePredictor(),
        config=GRPOTrainerConfig(group_size=4, prompts_per_step=2, max_length=32,
                                 device="cpu", seed=0),
    )
    trainer.step(0)
    assert all(torch.equal(a, b)
               for a, b in zip(ref_before, reference.parameters(), strict=True))


def test_step_is_deterministic_under_a_seed(monkeypatch):
    """Verified directly against the mutant this test guards: replacing
    `np.random.default_rng` with a seed-ignoring stub must make it FAIL (see
    task-7-review.md finding 3). The brief's own comparison (`reward_mean`)
    cannot do that on its own: a fresh, untrained policy structurally gates
    every candidate it samples, so `reward_mean == 0.0` on both sides
    regardless of any seeding.

    Two things make this non-degenerate: (1) `torch.manual_seed` right before
    each `_tiny()` call, so the two runs build BIT-IDENTICAL policy weights --
    without it, nothing else seeds `torch.nn.init`'s global RNG draws, so the
    two runs' policies would differ from each other for reasons that have
    nothing to do with `GRPOTrainer`; (2) comparing the actually-generated
    `texts` (captured via a `sample_groups` spy), not just `reward_mean`, so a
    change to the rollout-seed derivation is visible even on a step whose
    reward happens to be zero either way.
    """
    import polyt5.rl.trainer as trainer_mod

    def run():
        torch.manual_seed(1234)
        policy, tok = _tiny()
        reference, _ = _tiny()
        reference.load_state_dict(policy.state_dict())

        captured_texts: list[list[str]] = []
        real_sample_groups = trainer_mod.sample_groups

        def spy(*args, **kwargs):
            batch = real_sample_groups(*args, **kwargs)
            captured_texts.append(batch.texts)
            return batch

        monkeypatch.setattr(trainer_mod, "sample_groups", spy)

        t = GRPOTrainer(policy=policy, reference=reference, tokenizer=tok,
                        arm=build_arm("accuracy"), predictor=_FakePredictor(),
                        config=GRPOTrainerConfig(group_size=4, prompts_per_step=2,
                                                 max_length=32, device="cpu", seed=11))
        stats = t.step(0)
        return stats, captured_texts[0]

    (stats_a, texts_a), (stats_b, texts_b) = run(), run()
    assert stats_a["reward_mean"] == stats_b["reward_mean"]
    assert texts_a == texts_b
    assert texts_a, "sanity: the spy must actually have captured something"


def test_targets_are_sensitive_to_the_seed_not_just_reproducible(monkeypatch):
    """Closes a gap the reproducibility test above cannot: a stub that
    IGNORES its seed argument but is otherwise internally deterministic
    (e.g. always behaves like ``np.random.default_rng(0)`` regardless of what
    seed was actually requested) would still pass a same-seed-called-twice
    comparison trivially, since it is a pure function of nothing. Only
    comparing two DIFFERENT seeds against the trainer's ACTUAL, unmodified
    ``step()`` can catch that class of bug -- this spies on the real
    ``targets`` kwarg ``step()`` passes to ``sample_groups`` and asserts it
    changes when ``config.seed`` does.
    """
    import polyt5.rl.trainer as trainer_mod

    policy, tok = _tiny()
    reference, _ = _tiny()
    reference.load_state_dict(policy.state_dict())

    real_sample_groups = trainer_mod.sample_groups

    def targets_for(seed):
        captured = {}

        def spy(*args, **kwargs):
            captured["targets"] = kwargs["targets"]
            return real_sample_groups(*args, **kwargs)

        monkeypatch.setattr(trainer_mod, "sample_groups", spy)
        t = GRPOTrainer(policy=policy, reference=reference, tokenizer=tok,
                        arm=build_arm("accuracy"), predictor=_FakePredictor(),
                        config=GRPOTrainerConfig(group_size=1, prompts_per_step=4,
                                                 max_length=8, device="cpu", seed=seed))
        t.step(0)
        return captured["targets"]

    assert targets_for(11) != targets_for(12)


def test_reward_hacking_canary_is_suppressed_by_the_confidence_gate():
    """A predictor that is confidently wrong vs one that is uncertain.

    Both claim a perfect hit. The gated reward must prefer the confident one,
    which is the entire purpose of the confidence weighting.
    """
    from polyt5.rewards import build_arm as _build

    arm = _build("accuracy")
    confident = arm(["[At][C][C][O][At]"], [500.0], [(500.0, 2.0, 4)])[0].value
    uncertain = arm(["[At][C][C][O][At]"], [500.0], [(500.0, 45.0, 4)])[0].value
    assert confident > 2 * uncertain


def test_logs_weighted_and_unweighted_reward():
    policy, tok = _tiny()
    reference, _ = _tiny()
    trainer = GRPOTrainer(policy=policy, reference=reference, tokenizer=tok,
                          arm=build_arm("accuracy"), predictor=_FakePredictor(),
                          config=GRPOTrainerConfig(group_size=4, prompts_per_step=2,
                                                   max_length=32, device="cpu", seed=0))
    stats = trainer.step(0)
    for key in ("reward_mean", "reward_unweighted_mean", "kl", "clip_fraction",
                "gated_fraction", "mean_length"):
        assert key in stats, key


# -- additional coverage: mutant-check requirements from the task-7 brief --
# rather than in the literal Step 1 test block. See task-7-report.md for the
# rationale and the explicit red-step-against-each-mutant evidence.


def test_advantages_are_computed_per_group_not_over_the_flat_batch(monkeypatch):
    """Mutant (a): advantages computed over the flat batch instead of per group.

    Spies on the module-level ``group_advantages`` call trainer.py makes and
    asserts it is invoked with ``group_size=config.group_size`` (4), not 1
    (flat/no grouping) and not the full batch size (8, which would collapse
    grouping into a single group spanning two different prompts).
    """
    import polyt5.rl.trainer as trainer_mod

    policy, tok = _tiny()
    reference, _ = _tiny()
    reference.load_state_dict(policy.state_dict())

    seen_group_sizes: list[int] = []
    real_group_advantages = trainer_mod.group_advantages

    def spy(rewards, group_size, **kwargs):
        seen_group_sizes.append(group_size)
        return real_group_advantages(rewards, group_size, **kwargs)

    monkeypatch.setattr(trainer_mod, "group_advantages", spy)

    trainer = GRPOTrainer(
        policy=policy, reference=reference, tokenizer=tok,
        arm=build_arm("accuracy"), predictor=_FakePredictor(),
        config=GRPOTrainerConfig(group_size=4, prompts_per_step=2, max_length=16,
                                 device="cpu", seed=0),
    )
    trainer.step(0)
    assert seen_group_sizes == [4], seen_group_sizes


def test_policy_logprobs_carry_gradient_and_are_not_the_reference_logprobs(monkeypatch):
    """Mutant (b): reference log-probs used in place of the policy's own (or
    vice versa). Also covers mutant (mask): ``grpo_loss`` receiving something
    other than the real ``RolloutBatch.mask`` (see task-7-review.md finding 5
    -- ``mask=torch.ones_like(batch.mask)`` is a same-shape substitute that a
    values-only check could miss on a rollout that happens not to hit an
    early EOS, so this asserts object IDENTITY against the mask a
    ``sample_groups`` spy actually produced).

    Spies on the module-level ``grpo_loss`` call and asserts: the recomputed
    policy ``logprobs`` require grad (the whole point of "recompute under the
    policy WITH gradients"); ``old_logprobs`` and ``ref_logprobs`` do NOT
    (both come from no-grad paths -- rollout sampling and
    ``ReferencePolicy.score``); ``logprobs`` is not the same tensor object as
    ``ref_logprobs`` (a swap would hand the reference's detached tensor to the
    slot that must carry gradient, which the ``requires_grad`` check alone
    would also catch, but the identity check makes the swap direction
    unambiguous); and the ``mask`` argument IS the batch's own mask, by
    identity -- trainer.py passes ``batch.mask`` straight through by
    reference, never copying or reconstructing it.
    """
    import polyt5.rl.trainer as trainer_mod

    policy, tok = _tiny()
    reference, _ = _tiny()
    reference.load_state_dict(policy.state_dict())

    captured_batch = {}
    real_sample_groups = trainer_mod.sample_groups

    def sample_groups_spy(*args, **kwargs):
        batch = real_sample_groups(*args, **kwargs)
        captured_batch["batch"] = batch
        return batch

    monkeypatch.setattr(trainer_mod, "sample_groups", sample_groups_spy)

    captured = {}
    real_grpo_loss = trainer_mod.grpo_loss

    def spy(logprobs, old_logprobs, ref_logprobs, advantages, mask, *, config):
        captured["logprobs_requires_grad"] = logprobs.requires_grad
        captured["old_logprobs_requires_grad"] = old_logprobs.requires_grad
        captured["ref_logprobs_requires_grad"] = ref_logprobs.requires_grad
        captured["logprobs_is_ref_logprobs"] = logprobs is ref_logprobs
        captured["logprobs_is_old_logprobs"] = logprobs is old_logprobs
        captured["mask_is_batch_mask"] = mask is captured_batch["batch"].mask
        return real_grpo_loss(logprobs, old_logprobs, ref_logprobs, advantages, mask,
                              config=config)

    monkeypatch.setattr(trainer_mod, "grpo_loss", spy)

    trainer = GRPOTrainer(
        policy=policy, reference=reference, tokenizer=tok,
        arm=build_arm("accuracy"), predictor=_FakePredictor(),
        config=GRPOTrainerConfig(group_size=4, prompts_per_step=2, max_length=16,
                                 device="cpu", seed=0),
    )
    trainer.step(0)
    assert captured["logprobs_requires_grad"] is True
    assert captured["old_logprobs_requires_grad"] is False
    assert captured["ref_logprobs_requires_grad"] is False
    assert captured["logprobs_is_ref_logprobs"] is False
    assert captured["logprobs_is_old_logprobs"] is False
    assert captured["mask_is_batch_mask"] is True


def test_optimizer_step_is_actually_called(monkeypatch):
    """Mutant (c): optimizer.step() never called.

    A gradient-populated-but-never-applied bug is indistinguishable from a
    correct run by inspecting parameters alone if a later step happens to move
    them for an unrelated reason, so this spies directly on the bound
    ``optimizer.step`` method and counts calls.
    """
    policy, tok = _tiny()
    reference, _ = _tiny()
    reference.load_state_dict(policy.state_dict())

    trainer = GRPOTrainer(
        policy=policy, reference=reference, tokenizer=tok,
        arm=build_arm("accuracy"), predictor=_FakePredictor(),
        config=GRPOTrainerConfig(group_size=4, prompts_per_step=2, max_length=16,
                                 device="cpu", seed=0),
    )
    calls = {"n": 0}
    real_step = trainer.optimizer.step

    def counting_step(*args, **kwargs):
        calls["n"] += 1
        return real_step(*args, **kwargs)

    monkeypatch.setattr(trainer.optimizer, "step", counting_step)
    trainer.step(0)
    assert calls["n"] == 1


def test_gated_candidates_contribute_exactly_the_arms_own_gated_reward(monkeypatch):
    """Mutant (d): gated candidates still contributing their (ungated) reward.

    Forces one candidate in the group to be a real, valid polymer and the
    other to be structurally invalid garbage, then checks the trainer's
    ``reward_mean`` and ``gated_fraction`` against values computed by calling
    the SAME arm directly on the SAME (texts, targets, predictions) outside
    the trainer. A trainer that bypassed the arm's own gating (e.g. scored
    every candidate from the raw predictor mean regardless of structural
    validity) would diverge from this independently-computed expectation.
    """
    import polyt5.rl.trainer as trainer_mod

    policy, tok = _tiny()
    reference, _ = _tiny()
    reference.load_state_dict(policy.state_dict())

    forced_texts = ["[At][C][C][O][At]", "not a polymer at all"]
    forced_targets = [500.0, 500.0]
    predictor = _FakePredictor(mean=500.0, std=2.0)

    real_sample_groups = trainer_mod.sample_groups

    def fake_sample_groups(*args, **kwargs):
        batch = real_sample_groups(*args, **kwargs)
        return dataclasses.replace(batch, texts=forced_texts, targets=forced_targets)

    monkeypatch.setattr(trainer_mod, "sample_groups", fake_sample_groups)

    arm = build_arm("accuracy")
    expected_predictions = predictor.predict_with_uncertainty(forced_texts)
    expected_results = arm(forced_texts, forced_targets, expected_predictions)
    expected_reward_mean = float(np.mean([r.value for r in expected_results]))
    expected_gated_fraction = float(np.mean([r.gated for r in expected_results]))
    assert expected_gated_fraction == 0.5, "test setup: exactly one candidate must gate"

    trainer = GRPOTrainer(
        policy=policy, reference=reference, tokenizer=tok,
        arm=build_arm("accuracy"), predictor=_FakePredictor(mean=500.0, std=2.0),
        config=GRPOTrainerConfig(group_size=2, prompts_per_step=1, max_length=16,
                                 device="cpu", seed=0),
    )
    stats = trainer.step(0)
    assert stats["gated_fraction"] == expected_gated_fraction
    assert stats["reward_mean"] == expected_reward_mean


def test_ratio_is_exactly_one_at_step_zero_when_policy_equals_reference():
    """The central invariant the eval()/eval() recompute design buys (see
    task-7-review.md finding 1): with exactly one gradient step per rollout,
    ``pi_theta`` and ``pi_theta_old`` are the SAME distribution at recompute
    time whenever ``reference`` starts out equal to ``policy``, so the
    importance ratio should be ~1.0 and clip_fraction ~0 -- not dropout noise
    masquerading as policy drift before any update has happened.

    Reinstating ``self.policy.train()`` before the recompute (the reverted
    mutant) must make this test fail: dropout would separate the two passes
    at identical parameters, driving `mean_ratio` away from 1.0 and clipping
    a large fraction of tokens for no algorithmic reason.
    """
    policy, tok = _tiny()
    reference, _ = _tiny()
    reference.load_state_dict(policy.state_dict())
    trainer = GRPOTrainer(
        policy=policy, reference=reference, tokenizer=tok,
        arm=build_arm("accuracy"), predictor=_FakePredictor(),
        config=GRPOTrainerConfig(group_size=4, prompts_per_step=2, max_length=32,
                                 device="cpu", seed=0),
    )
    stats = trainer.step(0)
    assert stats["clip_fraction"] < 1e-6, stats["clip_fraction"]
    assert abs(stats["mean_ratio"] - 1.0) < 1e-4, stats["mean_ratio"]
    assert stats["kl"] < 1e-6, stats["kl"]


def test_reference_must_be_a_separate_object_from_policy():
    """Finding 7: a named error instead of relying on the indirect
    ``loss.backward()`` failure that ``ReferencePolicy`` freezing a shared
    object would otherwise produce.
    """
    policy, tok = _tiny()
    with pytest.raises(ValueError, match="[Ss]eparate"):
        GRPOTrainer(
            policy=policy, reference=policy, tokenizer=tok,
            arm=build_arm("accuracy"), predictor=_FakePredictor(),
            config=GRPOTrainerConfig(group_size=4, prompts_per_step=2, max_length=16,
                                     device="cpu", seed=0),
        )


def test_decoder_start_token_id_mismatch_raises():
    """Finding 9: a tokenizer/model decoder-start disagreement must fail
    loudly, not silently desynchronise ``logprobs`` from ``old_logprobs`` by
    one token.
    """
    policy, tok = _tiny()
    reference, _ = _tiny()
    reference.load_state_dict(policy.state_dict())
    # Desynchronise the model's own idea of the start token from the
    # tokenizer's -- `_policy_logprobs` is required to notice.
    policy.config.decoder_start_token_id = policy.config.decoder_start_token_id + 1

    trainer = GRPOTrainer(
        policy=policy, reference=reference, tokenizer=tok,
        arm=build_arm("accuracy"), predictor=_FakePredictor(),
        config=GRPOTrainerConfig(group_size=4, prompts_per_step=2, max_length=16,
                                 device="cpu", seed=0),
    )
    with pytest.raises(ValueError, match="decoder_start_token_id"):
        trainer.step(0)


def test_rollout_batch_size_is_forwarded_as_chunk_size(monkeypatch):
    """Finding 6: ``rollout_batch_size`` must actually reach
    ``sample_groups`` rather than being silently ignored (and, worse,
    recorded into checkpoint provenance as if it had governed the run).
    """
    import polyt5.rl.trainer as trainer_mod

    policy, tok = _tiny()
    reference, _ = _tiny()
    reference.load_state_dict(policy.state_dict())

    captured = {}
    real_sample_groups = trainer_mod.sample_groups

    def spy(*args, **kwargs):
        captured["chunk_size"] = kwargs.get("chunk_size")
        return real_sample_groups(*args, **kwargs)

    monkeypatch.setattr(trainer_mod, "sample_groups", spy)

    trainer = GRPOTrainer(
        policy=policy, reference=reference, tokenizer=tok,
        arm=build_arm("accuracy"), predictor=_FakePredictor(),
        config=GRPOTrainerConfig(group_size=4, prompts_per_step=2, max_length=16,
                                 device="cpu", seed=0, rollout_batch_size=3),
    )
    trainer.step(0)
    assert captured["chunk_size"] == 3


def test_train_loop_always_checkpoints_the_final_step(tmp_path):
    """Finding 8: log and save cadence must be consistent, and the final step
    must always checkpoint even when ``max_steps`` is not a multiple of
    ``save_every`` -- otherwise the last weights a run produced are silently
    never saved.
    """
    from polyt5.utils import RunDirectory

    policy, tok = _tiny()
    reference, _ = _tiny()
    reference.load_state_dict(policy.state_dict())
    run_dir = RunDirectory.create(tmp_path, "grpo_final_step")

    trainer = GRPOTrainer(
        policy=policy, reference=reference, tokenizer=tok,
        arm=build_arm("accuracy"), predictor=_FakePredictor(),
        config=GRPOTrainerConfig(group_size=2, prompts_per_step=2, max_length=16,
                                 device="cpu", seed=0, max_steps=3, log_every=10,
                                 save_every=10),
        run_dir=run_dir,
    )
    trainer.train()
    checkpoints = list(run_dir.checkpoints.glob("*.pt"))
    assert len(checkpoints) == 1, checkpoints
    assert checkpoints[0].name == "step_000003.pt", checkpoints[0].name


def test_train_loop_logs_and_checkpoints(tmp_path):
    """``.train()`` drives ``.step()`` for ``max_steps``, logging through
    ``RunDirectory.log_metrics`` and checkpointing every ``save_every`` steps.
    """
    from polyt5.utils import RunDirectory

    policy, tok = _tiny()
    reference, _ = _tiny()
    reference.load_state_dict(policy.state_dict())
    run_dir = RunDirectory.create(tmp_path, "grpo_smoke")

    trainer = GRPOTrainer(
        policy=policy, reference=reference, tokenizer=tok,
        arm=build_arm("accuracy"), predictor=_FakePredictor(),
        config=GRPOTrainerConfig(group_size=2, prompts_per_step=2, max_length=16,
                                 device="cpu", seed=0, max_steps=2, log_every=1,
                                 save_every=1),
        run_dir=run_dir,
    )
    result = trainer.train()
    assert isinstance(result, dict)
    assert run_dir.metrics_jsonl.exists()
    checkpoints = list(run_dir.checkpoints.glob("*.pt"))
    assert len(checkpoints) == 2, checkpoints
