"""Spec 4.4 drift monitoring, and the in-flight diagnostics that make a
degenerate optimum visible at step 50 instead of at hour 7.

The load-bearing test here is
:func:`test_attaching_a_drift_monitor_does_not_change_the_step_at_all`: the
monitor may carry the held-out auditor, and the whole study rests on split 4
never entering a reward path. That test is the structural guarantee -- rewards,
advantages and loss must be bit-identical with and without it.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.rewards import TgRewardConfig, build_arm
from polyt5.rl.drift import NEAR_COPY_THRESHOLD, DriftMonitor
from polyt5.rl.trainer import GRPOTrainer, GRPOTrainerConfig
from polyt5.tokenization import PolyT5Tokenizer

VALID = "[At][C][C][O][At]"
POLYETHYLENE_OXIDE = "[At]CCO[At]"


class _FakePredictor:
    """A four-member ensemble stand-in, matching the real triple contract."""

    def __init__(self, mean=500.0, std=5.0, n=4):
        self.mean, self.std, self.n = mean, std, n

    def __len__(self):
        return 4

    def predict_with_uncertainty(self, candidates):
        return [(self.mean, self.std, self.n) for _ in candidates]


def _tiny():
    tok = PolyT5Tokenizer.default()
    cfg = PolyT5Config(vocab_size=tok.vocab_size, d_model=64, d_kv=16, num_heads=4,
                       d_ff=128, num_layers=2, n_positions=64,
                       pad_token_id=tok.pad_id, eos_token_id=tok.eos_id,
                       decoder_start_token_id=tok.decoder_start_token_id)
    return PolyT5ForConditionalGeneration(cfg), tok


def _trainer(*, arm_name="accuracy", drift_monitor=None, predictor=None, **arm_kwargs):
    torch.manual_seed(4321)
    policy, tok = _tiny()
    reference, _ = _tiny()
    reference.load_state_dict(policy.state_dict())
    kwargs = {"ensemble_size": 4, **arm_kwargs}
    return GRPOTrainer(
        policy=policy, reference=reference, tokenizer=tok,
        arm=build_arm(arm_name, **kwargs),
        predictor=predictor or _FakePredictor(),
        config=GRPOTrainerConfig(group_size=4, prompts_per_step=2, max_length=16,
                                 device="cpu", seed=0),
        drift_monitor=drift_monitor,
    )


# --------------------------------------------------------------- DriftMonitor


def test_auditor_gap_is_measured_against_the_ensemble_mean():
    monitor = DriftMonitor(auditor=lambda candidates: [410.0, 380.0])
    stats = monitor.observe(["a", "b"], [(400.0, 3.0, 4), (400.0, 3.0, 4)])
    assert stats["auditor_gap_mean"] == pytest.approx(15.0)
    assert stats["auditor_gap_signed_mean"] == pytest.approx(-5.0)
    assert stats["auditor_gap_n"] == 2.0


def test_auditor_gap_skips_candidates_neither_side_could_score():
    nan = float("nan")
    monitor = DriftMonitor(auditor=lambda candidates: [nan, 380.0])
    stats = monitor.observe(["a", "b"], [(400.0, 3.0, 4), (nan, nan, 0)])
    assert stats["auditor_gap_n"] == 0.0
    assert stats["auditor_gap_mean"] is None


def test_a_monitor_without_an_auditor_reports_none_not_zero():
    monitor = DriftMonitor()
    stats = monitor.observe(["a"], [(400.0, 3.0, 4)])
    assert monitor.has_auditor is False
    assert stats["auditor_gap_mean"] is None
    assert stats["max_tanimoto_mean"] is None


def test_an_auditor_that_raises_does_not_kill_the_step():
    def exploding(_candidates):
        raise RuntimeError("predictor died")

    monitor = DriftMonitor(auditor=exploding)
    stats = monitor.observe(["a"], [(400.0, 3.0, 4)])
    assert stats["auditor_gap_n"] == 0.0


@pytest.mark.skipif(
    not __import__("polyt5.evaluation.similarity", fromlist=["x"]).FINGERPRINTS_AVAILABLE,
    reason="RDKit Morgan fingerprints unavailable in this environment",
)
def test_max_tanimoto_catches_the_near_copy_exact_match_novelty_cannot():
    """The reason this half exists.

    Novelty as the reward measures it is absence of the EXACT canonical form,
    so a memorised training polymer scores ``novel = 1.0`` after a one-atom
    edit. Here the reference set contains the polymer itself, and the monitor
    reports a max-Tanimoto of 1.0 -- the only in-flight signal that the
    "novel" chemistry is a copy.
    """
    monitor = DriftMonitor(reference_psmiles=[POLYETHYLENE_OXIDE])
    assert monitor.has_similarity
    stats = monitor.observe([VALID], [(400.0, 3.0, 4)])
    assert stats["max_tanimoto_n"] == 1.0
    assert stats["max_tanimoto_mean"] == pytest.approx(1.0)
    assert stats["near_copy_fraction"] == pytest.approx(1.0)
    assert NEAR_COPY_THRESHOLD <= 1.0


def test_monitor_never_exposes_the_auditor_it_holds():
    """Containment, structurally: there is no accessor that hands the
    held-out predictor back out to anything that could use it as a reward.
    """
    sentinel = lambda candidates: [1.0] * len(candidates)  # noqa: E731
    monitor = DriftMonitor(auditor=sentinel)
    public = {name: getattr(monitor, name) for name in dir(monitor)
              if not name.startswith("_")}
    assert sentinel not in public.values(), public.keys()


# ---------------------------------------- the containment guarantee, end to end


def test_attaching_a_drift_monitor_does_not_change_the_step_at_all():
    """The auditor is loaded into the training process for spec 4.4's gap log.
    Nothing about the reward, the advantages or the loss may depend on it.
    """
    seen: list[int] = []

    def recording_auditor(candidates):
        seen.append(len(candidates))
        return [123.0] * len(candidates)

    plain = _trainer().step(0)
    monitored = _trainer(
        drift_monitor=DriftMonitor(auditor=recording_auditor)
    ).step(0)

    assert seen, "the monitor did not actually run -- this test would prove nothing"
    for key in ("reward_mean", "reward_std", "loss", "policy_loss", "kl", "clip_fraction",
                "mean_ratio", "gated_fraction", "nonzero_advantage_fraction"):
        assert plain[key] == monitored[key], key


def test_drift_keys_appear_only_on_measured_steps():
    monitor = DriftMonitor(auditor=lambda candidates: [123.0] * len(candidates))
    trainer = _trainer(drift_monitor=monitor)
    trainer.config = dataclasses.replace(trainer.config, drift_every=3)
    assert "drift_auditor_gap_mean" in trainer.step(0)
    assert "drift_auditor_gap_mean" not in trainer.step(1)
    assert "drift_auditor_gap_mean" in trainer.step(3)


# ------------------------------------------------ in-flight diagnostics (S6/S7/S8)


class _Index:
    """Novelty index stand-in; ``known`` canonical forms are not novel."""

    def __init__(self, known=()):
        self._known = set(known)

    def is_novel(self, psmiles):
        return psmiles not in self._known


def _force_texts(monkeypatch, texts):
    """Make the rollout return exactly ``texts``, so a step is non-degenerate."""
    import polyt5.rl.trainer as trainer_mod

    real_sample_groups = trainer_mod.sample_groups

    def forced(*args, **kwargs):
        batch = real_sample_groups(*args, **kwargs)
        n = len(batch.texts)
        return dataclasses.replace(batch, texts=[texts[i % len(texts)] for i in range(n)])

    monkeypatch.setattr(trainer_mod, "sample_groups", forced)


def test_reward_unweighted_mean_is_none_for_an_arm_with_no_closeness_term(monkeypatch):
    """Finding S6: ``components["closeness"]`` is produced only by the two
    Tg-scoring arms, so the old ``.get("closeness", 0.0)`` logged an identical
    ``0.0`` for all 2000 steps of the validity and constraint runs -- a
    constant that reads exactly like a bug rather than a measurement.
    """
    _force_texts(monkeypatch, [VALID])
    accuracy = _trainer().step(0)
    validity = _trainer(arm_name="validity", novelty_index=_Index()).step(0)

    assert accuracy["reward_unweighted_mean"] is not None
    assert validity["reward_unweighted_mean"] is None, (
        "validity has no closeness term; reporting 0.0 claims a measurement that was "
        "never made"
    )


def test_step_stats_expose_the_partial_ensemble_counters(monkeypatch):
    """Finding B1's in-flight detector: a policy walking off the reward
    ensemble's support shows up here long before compare_arms runs.
    """
    _force_texts(monkeypatch, [VALID])
    full = _trainer(predictor=_FakePredictor(n=4)).step(0)
    partial = _trainer(predictor=_FakePredictor(n=1)).step(0)

    assert full["ensemble_size"] == 4.0
    assert full["ensemble_full_fraction"] == pytest.approx(1.0)
    assert full["ensemble_partial_fraction"] == pytest.approx(0.0)
    assert partial["ensemble_partial_fraction"] == pytest.approx(1.0)
    assert partial["mean_contributing_members"] == pytest.approx(1.0)
    # And the reward really is discounted, not merely counted.
    assert partial["reward_mean"] < full["reward_mean"]


def test_step_stats_expose_the_cascade_rates_the_arm_measured(monkeypatch):
    """Finding S7: ``results`` carried rich per-candidate components and every
    one was discarded except ``closeness``.
    """
    _force_texts(monkeypatch, [VALID])
    stats = _trainer(arm_name="validity", novelty_index=_Index()).step(0)
    for key in ("sv_pass_rate", "tsd_pass_rate", "dd_pass_rate", "pv_pass_rate"):
        assert stats[key] is not None, key
    # An arm that never produces the component reports None, never 0.0.
    assert stats["closeness_mean"] is None


def test_cross_group_dedup_kills_most_of_the_step_while_reward_mean_looks_fine(monkeypatch):
    """Finding S8. ``ValidityArm``'s ``seen_canonical`` set is per-CALL and the
    trainer calls the arm once on the whole step, so DD dedups ACROSS groups.
    Under mode collapse only the first group scores at all; every later group
    has zero reward variance and therefore exactly zero advantage, and even
    the first group's advantages sum to zero over identical sequences. Nothing
    in ``reward_mean`` says so.
    """
    _force_texts(monkeypatch, [VALID])
    stats = _trainer(arm_name="validity", novelty_index=_Index()).step(0)

    assert stats["unique_fraction"] == pytest.approx(1 / 8), "the collapse itself is visible"
    assert stats["reward_mean"] > 0.0, "reward_mean still reads as if something happened"
    assert stats["zero_variance_group_fraction"] >= 0.5, (
        "every group after the first is dead and nothing reported it")
    assert stats["nonzero_advantage_fraction"] <= 0.5


def test_a_wholly_zero_gradient_step_is_reported_loudly(caplog, monkeypatch):
    """The fully degenerate case: every candidate identical AND already in the
    reference index, so every reward is 0.0, every advantage is exactly 0.0,
    and the only thing that moved the policy was the KL anchor.
    """
    from polyt5.chemistry.canonicalization import canonical_psmiles
    from polyt5.chemistry.conversion import pselfies_to_psmiles

    known = canonical_psmiles(pselfies_to_psmiles(VALID))
    _force_texts(monkeypatch, [VALID])
    with caplog.at_level("WARNING", logger="polyt5.rl.trainer"):
        stats = _trainer(arm_name="validity", novelty_index=_Index([known])).step(0)

    assert stats["reward_mean"] == pytest.approx(0.0)
    assert stats["nonzero_advantage_fraction"] == pytest.approx(0.0)
    assert stats["zero_variance_group_fraction"] == pytest.approx(1.0)
    assert stats["tsd_pass_rate"] == pytest.approx(0.0), "the collapse is attributable to TSD"
    assert any("advantage" in record.message.lower() for record in caplog.records), (
        [record.message for record in caplog.records]
    )


def test_a_mostly_partial_ensemble_is_reported_loudly(caplog, monkeypatch):
    _force_texts(monkeypatch, [VALID])
    with caplog.at_level("WARNING", logger="polyt5.rl.trainer"):
        _trainer(predictor=_FakePredictor(n=1)).step(0)
    assert any("part of the property ensemble" in record.message.lower()
               for record in caplog.records), [record.message for record in caplog.records]


def test_reward_std_is_reported_alongside_the_mean(monkeypatch):
    _force_texts(monkeypatch, [VALID, "not a polymer"])
    stats = _trainer().step(0)
    assert np.isfinite(stats["reward_std"])
    assert stats["reward_std"] > 0.0, "half the batch is gated; the spread cannot be zero"


# ----------------------------------------------------- reference policy (minor 14)


def test_reference_policy_checks_the_tokenizer_start_token():
    """Minor 14: Task 7's rule -- "read the start token from the tokenizer in
    both, and assert the model config agrees" -- was applied to the trainer and
    to ``sample_groups`` but not to this, the third consumer.
    """
    from polyt5.rl.reference_policy import ReferencePolicy

    model, tok = _tiny()
    model.config.decoder_start_token_id = model.config.decoder_start_token_id + 1
    with pytest.raises(ValueError, match="decoder_start_token_id"):
        ReferencePolicy(model, device="cpu", tokenizer=tok)


def test_reference_policy_still_works_without_a_tokenizer():
    from polyt5.rl.reference_policy import ReferencePolicy

    model, tok = _tiny()
    policy = ReferencePolicy(model, device="cpu")
    assert policy.decoder_start_token_id == tok.decoder_start_token_id


def test_trainer_passes_its_tokenizer_to_the_reference():
    policy, tok = _tiny()
    reference, _ = _tiny()
    reference.config.decoder_start_token_id = reference.config.decoder_start_token_id + 1
    with pytest.raises(ValueError, match="decoder_start_token_id"):
        GRPOTrainer(
            policy=policy, reference=reference, tokenizer=tok,
            arm=build_arm("accuracy", ensemble_size=4,
                          tg_config=TgRewardConfig()),
            predictor=_FakePredictor(),
            config=GRPOTrainerConfig(group_size=2, prompts_per_step=1, max_length=8,
                                     device="cpu", seed=0),
        )
