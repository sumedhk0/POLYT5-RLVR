# tests/test_rl_trainer.py
from __future__ import annotations

import dataclasses

import numpy as np
import torch

from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.rewards import build_arm
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


def test_one_step_runs_end_to_end_and_changes_parameters():
    policy, tok = _tiny()
    reference, _ = _tiny()
    reference.load_state_dict(policy.state_dict())
    before = [p.detach().clone() for p in policy.parameters()]
    trainer = GRPOTrainer(
        policy=policy, reference=reference, tokenizer=tok,
        arm=build_arm("accuracy"), predictor=_FakePredictor(),
        config=GRPOTrainerConfig(group_size=4, prompts_per_step=2, max_length=32,
                                 device="cpu", seed=0, learning_rate=1e-3),
    )
    stats = trainer.step(0)
    assert np.isfinite(stats["reward_mean"])
    assert np.isfinite(stats["loss"])
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


def test_step_is_deterministic_under_a_seed():
    def run():
        policy, tok = _tiny()
        reference, _ = _tiny()
        reference.load_state_dict(policy.state_dict())
        t = GRPOTrainer(policy=policy, reference=reference, tokenizer=tok,
                        arm=build_arm("accuracy"), predictor=_FakePredictor(),
                        config=GRPOTrainerConfig(group_size=4, prompts_per_step=2,
                                                 max_length=32, device="cpu", seed=11))
        return t.step(0)
    a, b = run(), run()
    assert a["reward_mean"] == b["reward_mean"]


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
    vice versa).

    Spies on the module-level ``grpo_loss`` call and asserts: the recomputed
    policy ``logprobs`` require grad (the whole point of "recompute under the
    policy WITH gradients"); ``old_logprobs`` and ``ref_logprobs`` do NOT
    (both come from no-grad paths -- rollout sampling and
    ``ReferencePolicy.score``); and ``logprobs`` is not the same tensor object
    as ``ref_logprobs`` (a swap would hand the reference's detached tensor to
    the slot that must carry gradient, which the ``requires_grad`` check alone
    would also catch, but the identity check makes the swap direction
    unambiguous).
    """
    import polyt5.rl.trainer as trainer_mod

    policy, tok = _tiny()
    reference, _ = _tiny()
    reference.load_state_dict(policy.state_dict())

    captured = {}
    real_grpo_loss = trainer_mod.grpo_loss

    def spy(logprobs, old_logprobs, ref_logprobs, advantages, mask, *, config):
        captured["logprobs_requires_grad"] = logprobs.requires_grad
        captured["old_logprobs_requires_grad"] = old_logprobs.requires_grad
        captured["ref_logprobs_requires_grad"] = ref_logprobs.requires_grad
        captured["logprobs_is_ref_logprobs"] = logprobs is ref_logprobs
        captured["logprobs_is_old_logprobs"] = logprobs is old_logprobs
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
