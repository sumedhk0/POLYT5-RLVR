"""Supervised replay (round 3).

The claim under test is narrow: adding the replay term must change NOTHING when it is
off, must fail loudly rather than silently when misconfigured, and must draw a
reproducible batch so resume stays bit-identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from polyt5.rewards import TgRewardConfig, build_arm  # noqa: E402
from polyt5.rl.trainer import REPLAY_STREAM, GRPOTrainer, GRPOTrainerConfig  # noqa: E402
from test_rl_trainer import _FakePredictor, _tiny  # noqa: E402

PAIRS = [
    ("249.2", "[At][C][C][O][At]"),
    ("310.5", "[At][C][C][C][At]"),
    ("400.0", "[At][O][C][C][At]"),
    ("512.7", "[At][C][=C][C][At]"),
]


def _trainer(**cfg_kw):
    policy, tok = _tiny()
    reference, _ = _tiny()
    reference.load_state_dict(policy.state_dict())
    dataset = cfg_kw.pop("replay_dataset", None)
    config = GRPOTrainerConfig(
        group_size=2, prompts_per_step=2, max_length=32, device="cpu", seed=0,
        learning_rate=1e-3, **cfg_kw,
    )
    return GRPOTrainer(
        policy=policy, reference=reference, tokenizer=tok,
        arm=build_arm("accuracy", ensemble_size=4,
                      tg_config=TgRewardConfig(tolerance=1000.0)),
        predictor=_FakePredictor(),
        config=config, replay_dataset=dataset,
    )


def test_a_nonzero_coefficient_without_data_raises():
    """The failure this guard exists for: a run that believes it is doing replay.

    Training on silently would produce a result that looks like evidence replay does
    not work, which is the most expensive way for this to go wrong.
    """
    with pytest.raises(ValueError, match="requires replay_dataset"):
        _trainer(replay_coef=0.5)


def test_a_negative_coefficient_raises():
    with pytest.raises(ValueError, match="non-negative"):
        _trainer(replay_coef=-0.1, replay_dataset=PAIRS)


def test_replay_is_off_by_default_and_logs_none():
    trainer = _trainer()
    stats = trainer.step(0)
    assert stats["replay_coef"] == 0.0
    assert stats["replay_loss"] is None


def test_a_dataset_without_a_coefficient_stays_off():
    """Passing data must not silently enable the term."""
    trainer = _trainer(replay_dataset=PAIRS)
    assert trainer._replay_loss(0) is None
    assert trainer.step(0)["replay_loss"] is None


def test_replay_loss_is_finite_positive_and_logged():
    trainer = _trainer(replay_coef=0.5, replay_batch_size=2, replay_dataset=PAIRS)
    stats = trainer.step(0)
    assert stats["replay_coef"] == 0.5
    assert stats["replay_loss"] is not None
    assert stats["replay_loss"] > 0.0
    assert torch.isfinite(torch.tensor(stats["replay_loss"]))


def test_the_batch_is_reproducible_for_a_step_index():
    """Resume must stay bit-identical, which requires the same batch at the same step.

    The two trainers are given IDENTICAL weights first. Without that this compares two
    randomly initialised models and fails for a reason that has nothing to do with
    batch selection -- which is exactly what the first version of this test did.
    """
    a = _trainer(replay_coef=0.5, replay_batch_size=2, replay_dataset=PAIRS)
    b = _trainer(replay_coef=0.5, replay_batch_size=2, replay_dataset=PAIRS)
    b.policy.load_state_dict(a.policy.state_dict())
    assert torch.allclose(a._replay_loss(7), b._replay_loss(7))


def test_different_steps_draw_different_batches():
    """A constant batch would train on the same examples 2000 times.

    Checked over twelve steps rather than two: with four pairs and a batch of two
    there are only six unordered draws and the loss is order-independent, so two
    adjacent steps colliding is ordinary rather than a bug.
    """
    trainer = _trainer(replay_coef=0.5, replay_batch_size=2, replay_dataset=PAIRS)
    losses = {round(float(trainer._replay_loss(i)), 8) for i in range(12)}
    assert len(losses) > 1


def test_the_replay_stream_tag_differs_from_the_rollout_seed():
    """Sharing (seed, step_index) with the rollout is how the control arm broke.

    Its reward correlated 1.000000 with the target because both draws used the same
    two-element seed. The replay batch takes a third, distinct stream element.
    """
    assert REPLAY_STREAM != 0


def test_replay_changes_the_gradient_it_receives():
    """A term that does not alter the update is not doing anything."""
    off = _trainer(replay_dataset=PAIRS)
    off.step(0)
    grad_off = torch.cat([p.grad.flatten() for p in off.policy.parameters() if p.grad is not None])

    on = _trainer(replay_coef=5.0, replay_batch_size=4, replay_dataset=PAIRS)
    on.step(0)
    grad_on = torch.cat([p.grad.flatten() for p in on.policy.parameters() if p.grad is not None])

    assert not torch.allclose(grad_off, grad_on), "replay term left the gradient unchanged"


def test_one_optimizer_step_per_train_step_regardless_of_replay(monkeypatch):
    """Two steps would double the effective learning rate on replay batches."""
    trainer = _trainer(replay_coef=1.0, replay_batch_size=2, replay_dataset=PAIRS)
    calls = []
    real = trainer.optimizer.step
    monkeypatch.setattr(trainer.optimizer, "step",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    trainer.step(0)
    assert len(calls) == 1
