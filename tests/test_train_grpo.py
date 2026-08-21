"""Unit coverage for ``scripts/train_grpo.py``'s pure-function internals.

None of these need a model, a checkpoint, or a GPU: ``verify_artifact`` and
``load_frozen_baseline`` operate on small local files this module writes
itself, ``build_reward_ensemble``'s auditor guard and ``build_reward_arm``'s
config wiring are pure Python over ``polyt5.rewards`` objects (torch-free),
and ``_latest_checkpoint`` only touches the filesystem.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_grpo  # noqa: E402
from train_grpo import (  # noqa: E402
    _latest_checkpoint,
    build_drift_monitor,
    build_reward_arm,
    build_reward_ensemble,
    check_reward_overrides,
    load_frozen_baseline,
    parse_args,
    sha256_of_file,
    summarize_reward_overrides,
    verify_artifact,
)

# ------------------------------------------------------------- sha256_of_file


def test_sha256_of_file_matches_hashlib(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"the quick brown fox" * 100)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha256_of_file(path) == expected


# --------------------------------------------------------------- verify_artifact


def test_verify_artifact_passes_on_matching_hash(tmp_path):
    path = tmp_path / "ckpt.pt"
    path.write_bytes(b"weights")
    verify_artifact(path, sha256_of_file(path), label="test")  # must not raise


def test_verify_artifact_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_artifact(tmp_path / "missing.pt", "0" * 64, label="test")


def test_verify_artifact_raises_on_hash_mismatch(tmp_path):
    """A tampered or wrong checkpoint must raise, not warn -- this is the
    guard that keeps RL from ever training against an unverified baseline
    artifact.
    """
    path = tmp_path / "ckpt.pt"
    path.write_bytes(b"weights")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_artifact(path, "0" * 64, label="test")


# ---------------------------------------------------------- load_frozen_baseline


def _minimal_frozen(**overrides):
    base = {
        "artifacts": {"tokenizer": {"path": "x", "sha256": "a" * 64}},
        "reward_ensemble": ["split0", "split1", "split2", "split3"],
        "auditor": "split4",
        "evaluation_protocol": {"targets_k": [300, 400, 500], "n_per_target": 500},
        "success_criterion": "beats arm_b and survives the auditor",
    }
    base.update(overrides)
    return base


def test_load_frozen_baseline_loads_a_well_formed_record(tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(_minimal_frozen()), encoding="utf-8")
    frozen = load_frozen_baseline(path)
    assert frozen["auditor"] == "split4"


def test_load_frozen_baseline_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_frozen_baseline(tmp_path / "does_not_exist.json")


@pytest.mark.parametrize("key", [
    "artifacts", "reward_ensemble", "auditor", "evaluation_protocol", "success_criterion",
])
def test_load_frozen_baseline_raises_on_missing_required_key(tmp_path, key):
    payload = _minimal_frozen()
    del payload[key]
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(KeyError):
        load_frozen_baseline(path)


def test_load_frozen_baseline_raises_when_auditor_leaks_into_ensemble(tmp_path):
    payload = _minimal_frozen(reward_ensemble=["split0", "split4"], auditor="split4")
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="auditor"):
        load_frozen_baseline(path)


# --------------------------------------------------------- build_reward_ensemble


def test_build_reward_ensemble_raises_when_auditor_in_ensemble_list():
    """The second, independent guard inside ``build_reward_ensemble`` itself
    -- fires before any checkpoint is touched, so no real ``.pt`` file is
    needed to exercise it.
    """
    frozen = _minimal_frozen(reward_ensemble=["split0", "split4"], auditor="split4")
    with pytest.raises(ValueError, match="auditor"):
        build_reward_ensemble(frozen, Path("unused"), device="cpu", batch_size=1, num_beams=1)


# -------------------------------------------------------------- build_reward_arm


def test_build_reward_arm_wires_composite_weights_from_config():
    cfg = {"reward": {"weights": {"tg": 0.5, "pv": 0.3, "novelty": 0.2}}}
    arm = build_reward_arm("composite", cfg, novelty_index=None)
    assert arm.weights == {"tg": 0.5, "pv": 0.3, "novelty": 0.2}


def test_build_reward_arm_splits_tolerance_and_window_tolerance():
    """``reward.tolerance`` feeds ``TgRewardConfig``; ``reward.window_tolerance``
    feeds the arm's own (``_BaseArm``) ``tolerance`` -- these must be
    independently settable, not one key wearing two hats (finding 12).
    """
    cfg = {"reward": {"tolerance": 123.0, "window_tolerance": 77.0, "sigma0": 9.0}}
    arm = build_reward_arm("constraint", cfg, novelty_index=None)
    assert arm.tg_config.tolerance == 123.0
    assert arm.tg_config.sigma0 == 9.0
    assert arm.tolerance == 77.0


def test_build_reward_arm_defaults_match_documented_values():
    arm = build_reward_arm("accuracy", {})
    assert arm.tg_config.tolerance == 100.0
    assert arm.tolerance == 50.0


def test_build_reward_arm_novelty_index_override_is_used_verbatim():
    class FakeIndex:
        def is_novel(self, psmiles):
            return True

    fake = FakeIndex()
    arm = build_reward_arm("composite", {}, novelty_index=fake)
    assert arm.novelty_index is fake


def test_build_reward_arm_explicit_none_means_no_index_not_a_lookup(tmp_path):
    """Passing ``novelty_index=None`` explicitly (e.g. ``compare_arms.py``
    under ``--allow-missing-novelty-index``) must NOT fall back to opening
    ``cfg``'s own path -- that path is exactly the one already known to be
    missing; falling back to it would raise ``FileNotFoundError`` right back.
    """
    cfg = {"reward": {"novelty_index": str(tmp_path / "does_not_exist")}}
    arm = build_reward_arm("composite", cfg, novelty_index=None)
    assert arm.novelty_index is None


def test_build_reward_arm_arm_not_needing_index_ignores_bogus_novelty_path():
    cfg = {"reward": {"novelty_index": "/does/not/exist"}}
    arm = build_reward_arm("accuracy", cfg)  # must not raise / must not try to open it
    assert arm.novelty_index is None


# ---------------------------------------------------------------- _latest_checkpoint


def test_latest_checkpoint_picks_highest_step(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for name in ("step_000001.pt", "step_000010.pt", "step_000002.pt"):
        (checkpoints / name).write_bytes(b"")
    assert _latest_checkpoint(tmp_path).name == "step_000010.pt"


def test_latest_checkpoint_accepts_a_checkpoints_dir_directly(tmp_path):
    (tmp_path / "step_000003.pt").write_bytes(b"")
    assert _latest_checkpoint(tmp_path).name == "step_000003.pt"


def test_latest_checkpoint_returns_a_file_path_verbatim(tmp_path):
    ckpt = tmp_path / "explicit.pt"
    ckpt.write_bytes(b"")
    assert _latest_checkpoint(ckpt) == ckpt


def test_latest_checkpoint_raises_when_nothing_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        _latest_checkpoint(tmp_path)


def test_latest_checkpoint_ignores_non_step_named_pt_files(tmp_path):
    """Finding 14: the glob is ``step_*.pt`` specifically, not ``*.pt`` -- an
    unrelated ``.pt`` file dropped in ``checkpoints/`` must not be picked up.
    """
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "best.pt").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        _latest_checkpoint(tmp_path)


# ------------------------------------------------- ensemble_size wiring (Ruling F)


def test_build_reward_arm_requires_the_caller_to_declare_the_ensemble_size():
    """Ruling F. ``n_contributing == 1`` means "full coverage" for a
    single-model predictor and "three of four members could not read this
    candidate" for the reward ensemble; only the declared size separates them,
    and it must come from the frozen record's ``reward_ensemble`` list (via
    ``len(predictor)``), never from an arm's own ``--set``-overridable config.
    """
    assert build_reward_arm("accuracy", {}).ensemble_size == 1
    assert build_reward_arm("accuracy", {}, ensemble_size=4).ensemble_size == 4


def test_build_reward_arm_ignores_a_config_supplied_ensemble_size():
    """A ``--set reward.ensemble_size=1`` on a four-member ensemble would
    silently restore the inverted confidence weight, so the config is not
    allowed to say.
    """
    cfg = {"reward": {"ensemble_size": 1}}
    assert build_reward_arm("accuracy", cfg, ensemble_size=4).ensemble_size == 4


def test_build_reward_arm_threads_the_new_tg_constants():
    cfg = {"reward": {"sigma_unknown": 30.0, "min_coverage": 0.75}}
    arm = build_reward_arm("accuracy", cfg, ensemble_size=4)
    assert arm.tg_config.sigma_unknown == 30.0
    assert arm.tg_config.min_coverage == 0.75


def test_build_reward_arm_tg_constant_defaults_match_the_documented_values():
    arm = build_reward_arm("accuracy", {})
    assert arm.tg_config.sigma_unknown == pytest.approx(45.2)
    assert arm.tg_config.min_coverage == pytest.approx(0.5)


# --------------------------------------------------- drift monitor (spec 4.4)
#
# Split 4 (the auditor) must be truly held out: absent from the training
# process entirely, not merely prevented from reaching a reward. The default
# path (``load_auditor=False``) must never open, verify, or construct the
# auditor checkpoint at all -- max-Tanimoto similarity monitoring stays on,
# since it needs no auditor and is not itself optimized against (unlike
# sigma, which the confidence weight rewards the policy for shrinking).


def _bogus_frozen(*, reward_ensemble=("split0", "split1", "split2", "split3")):
    """A ``frozen_baseline.json``-shaped dict whose auditor artifact path does
    not exist -- opening it would raise ``FileNotFoundError``, so any test
    that reaches real file access on it fails loudly, not silently.
    """
    return {
        "artifacts": {"split4": {"path": "nope.pt", "sha256": "0" * 64}},
        "reward_ensemble": list(reward_ensemble),
        "auditor": "split4",
    }


def test_build_drift_monitor_default_never_touches_the_auditor(tmp_path, monkeypatch):
    """The containment property itself: patch the two functions that would
    have to run to open the auditor checkpoint so each raises if called, then
    prove the default call completes without either firing.
    """
    def _must_not_verify(*_args, **_kwargs):
        raise AssertionError("verify_artifact must not run for the auditor by default")

    class _MustNotConstruct:
        @classmethod
        def from_checkpoint(cls, *_args, **_kwargs):
            raise AssertionError(
                "PolyT5PropertyPredictor must not be constructed for the auditor by default")

    monkeypatch.setattr(train_grpo, "verify_artifact", _must_not_verify)
    monkeypatch.setattr(train_grpo, "PolyT5PropertyPredictor", _MustNotConstruct)

    monitor = build_drift_monitor(
        _bogus_frozen(), {}, tmp_path / "tok.json", device="cpu",
        batch_size=1, num_beams=1, seed=0,
    )
    assert monitor.has_auditor is False


def test_build_drift_monitor_default_still_has_similarity(tmp_path):
    """The one signal that stays on by default: max-Tanimoto needs no
    predictor at all.
    """
    reference = tmp_path / "train.jsonl"
    reference.write_text(json.dumps({"target": "[At][C][C][O][At]"}) + "\n", encoding="utf-8")

    monitor = build_drift_monitor(
        _bogus_frozen(), {"drift": {"reference": str(reference)}}, tmp_path / "tok.json",
        device="cpu", batch_size=1, num_beams=1, seed=0,
    )
    assert monitor.has_auditor is False
    assert monitor.has_similarity is True


def test_build_drift_monitor_load_auditor_true_loads_and_verifies_it(tmp_path, monkeypatch):
    """``load_auditor=True`` (wired from ``--drift-auditor``) is the opt-in
    path: the checkpoint IS verified and constructed in this case.
    """
    verified: list[str] = []

    def _fake_verify(_path, _sha256, *, label):
        verified.append(label)

    class _FakePredictor:
        @classmethod
        def from_checkpoint(cls, *_args, **_kwargs):
            return lambda candidates: [0.0 for _ in candidates]

    monkeypatch.setattr(train_grpo, "verify_artifact", _fake_verify)
    monkeypatch.setattr(train_grpo, "PolyT5PropertyPredictor", _FakePredictor)

    monitor = build_drift_monitor(
        _bogus_frozen(), {}, tmp_path / "tok.json", device="cpu",
        batch_size=1, num_beams=1, seed=0, load_auditor=True,
    )
    assert monitor.has_auditor is True
    assert verified == ["split4"]


def test_build_drift_monitor_refuses_an_auditor_that_leaks_into_the_ensemble(tmp_path):
    """The third independent guard on the same invariant, placed at the only
    point in the TRAINING path that opens the auditor's checkpoint at all --
    so it only fires when the auditor is actually requested.
    """
    frozen = _bogus_frozen(reward_ensemble=["split4"])
    with pytest.raises(ValueError, match="auditor"):
        build_drift_monitor(frozen, {}, tmp_path / "tok.json", device="cpu",
                            batch_size=1, num_beams=1, seed=0, load_auditor=True)


def test_parse_args_drift_auditor_flag_defaults_off():
    assert parse_args(["--arm", "accuracy"]).drift_auditor is False
    assert parse_args(["--arm", "accuracy", "--drift-auditor"]).drift_auditor is True


def _stub_main_pipeline_up_to_drift_decision(monkeypatch):
    """Stub every real object ``main()`` builds up to (but not past) the
    drift-monitor decision, so a test can observe exactly what happens at
    that one decision point without a real checkpoint, model, or GPU.

    Returns the sentinel exception ``build_trainer_config`` (the first real
    call after the drift block) raises, so a test that never triggers the
    drift block can still prove it got that far.
    """
    frozen = _minimal_frozen()

    class _FakeTokenizer:
        sha256 = "deadbeef"

    class _FakePredictor:
        def __len__(self):
            return 4

    monkeypatch.setattr(train_grpo, "load_frozen_baseline", lambda _path: frozen)
    monkeypatch.setattr(train_grpo, "build_verified_tokenizer", lambda _frozen: _FakeTokenizer())
    monkeypatch.setattr(train_grpo, "select_device", lambda _device: "cpu")
    monkeypatch.setattr(train_grpo, "load_verified_model", lambda _frozen, _key: (object(), {}))
    monkeypatch.setattr(train_grpo, "build_reward_ensemble", lambda *a, **k: _FakePredictor())
    monkeypatch.setattr(train_grpo, "build_reward_arm", lambda *a, **k: object())

    sentinel = RuntimeError("stopped after the drift-monitor decision (by design)")

    def _stop(*_a, **_k):
        raise sentinel

    monkeypatch.setattr(train_grpo, "build_trainer_config", _stop)
    return sentinel


def test_main_no_drift_monitor_never_calls_build_drift_monitor(tmp_path, monkeypatch):
    _stub_main_pipeline_up_to_drift_decision(monkeypatch)

    def _must_not_be_called(*_a, **_k):
        raise AssertionError("build_drift_monitor must not run under --no-drift-monitor")

    monkeypatch.setattr(train_grpo, "build_drift_monitor", _must_not_be_called)

    with pytest.raises(RuntimeError, match="stopped after the drift-monitor decision"):
        train_grpo.main([
            "--arm", "accuracy", "--no-drift-monitor",
            "--set", f"output.results_root={tmp_path}",
        ])


def test_main_default_wires_load_auditor_false_into_build_drift_monitor(tmp_path, monkeypatch):
    sentinel = _stub_main_pipeline_up_to_drift_decision(monkeypatch)
    captured: dict[str, Any] = {}

    def _capture(*_a, **kwargs):
        captured.update(kwargs)
        raise sentinel

    monkeypatch.setattr(train_grpo, "build_drift_monitor", _capture)

    with pytest.raises(RuntimeError, match="stopped after the drift-monitor decision"):
        train_grpo.main(["--arm", "accuracy", "--set", f"output.results_root={tmp_path}"])
    assert captured["load_auditor"] is False


def test_main_drift_auditor_flag_wires_load_auditor_true_into_build_drift_monitor(
    tmp_path, monkeypatch
):
    sentinel = _stub_main_pipeline_up_to_drift_decision(monkeypatch)
    captured: dict[str, Any] = {}

    def _capture(*_a, **kwargs):
        captured.update(kwargs)
        raise sentinel

    monkeypatch.setattr(train_grpo, "build_drift_monitor", _capture)

    with pytest.raises(RuntimeError, match="stopped after the drift-monitor decision"):
        train_grpo.main([
            "--arm", "accuracy", "--drift-auditor",
            "--set", f"output.results_root={tmp_path}",
        ])
    assert captured["load_auditor"] is True


def test_load_drift_reference_reads_pselfies_and_drops_junk(tmp_path):
    from train_grpo import load_drift_reference

    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps({"source": "300.0", "target": "[At][C][C][O][At]"}) + "\n"
        + "\n"
        + json.dumps({"source": "300.0", "target": "[Zz][not-a-token]"}) + "\n"
        + json.dumps({"source": "300.0", "target": "[At][C][C][C][At]"}) + "\n",
        encoding="utf-8",
    )
    reference = load_drift_reference(path, limit=10)
    assert len(reference) == 2, reference
    assert all("At" in entry or "*" in entry for entry in reference)


def test_load_drift_reference_missing_file_is_empty_not_fatal(tmp_path):
    """A drift monitor is a diagnostic; a run must not die because one could
    not be fully built.
    """
    from train_grpo import load_drift_reference

    assert load_drift_reference(tmp_path / "absent.jsonl", limit=10) == []


def test_load_drift_reference_honours_the_cap(tmp_path):
    from train_grpo import load_drift_reference

    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(json.dumps({"target": "[At][C][C][O][At]"}) for _ in range(10)),
        encoding="utf-8",
    )
    assert len(load_drift_reference(path, limit=3)) == 3


def test_trainer_config_reads_drift_every_from_the_config():
    from train_grpo import build_trainer_config

    config = build_trainer_config({"drift": {"every": 7}}, seed=0, device_override="cpu")
    assert config.drift_every == 7
    assert build_trainer_config({}, seed=0, device_override="cpu").drift_every == 50


def test_every_shipped_arm_config_declares_a_drift_block():
    from polyt5.utils import load_config

    for arm in ("accuracy", "validity", "composite", "constraint"):
        cfg = load_config(REPO_ROOT / "configs" / "rl" / f"{arm}.yaml")
        assert "drift" in cfg, arm
        assert cfg["drift"]["enabled"] is True, arm
        assert int(cfg["drift"]["every"]) >= 1, arm


# ------------------------------------------------- pinned reward overrides (item 2)


@pytest.mark.parametrize("key,value", [
    ("sigma_unknown", 0.0),
    ("sigma0", 1.0),
    ("min_coverage", 0.0),
])
def test_check_reward_overrides_refuses_ruling_f_constants(key, value):
    """The exact scenario the review named: ``--set reward.sigma_unknown=0.0``
    would silently revert Ruling F (measured: a 1-of-4 candidate scores
    1.0000 with sigma_unknown=0.0 vs 0.0683 at the pinned value). Must be
    refused, not merely logged.
    """
    problems = check_reward_overrides({"reward": {key: value}})
    assert problems, (key, value)
    assert any(key in problem for problem in problems), problems


def test_check_reward_overrides_allows_unpinned_reward_keys():
    """tolerance/sa_max/window_tolerance/weights are design choices a
    legitimate ablation might sweep -- not refused, only the three Ruling F
    constants are.
    """
    overrides = {"reward": {"tolerance": 50.0, "sa_max": 4.0, "window_tolerance": 25.0,
                            "weights": {"tg": 2.0}}}
    assert check_reward_overrides(overrides) == []


def test_check_reward_overrides_allows_no_reward_block():
    assert check_reward_overrides({}) == []
    assert check_reward_overrides({"train": {"max_steps": 10}}) == []


def test_check_reward_overrides_reports_every_pinned_key_touched():
    problems = check_reward_overrides(
        {"reward": {"sigma0": 1.0, "sigma_unknown": 0.0, "min_coverage": 1.0}})
    assert len(problems) == 3


def test_summarize_reward_overrides_records_canonical_and_overridden_values():
    overrides = {"reward": {"tolerance": 50.0}}
    canonical_cfg = {"reward": {"tolerance": 100.0, "sa_max": 6.0}}
    record = summarize_reward_overrides(overrides, canonical_cfg)
    assert record == {"tolerance": {"canonical": 100.0, "overridden_to": 50.0}}


def test_summarize_reward_overrides_empty_when_no_reward_overrides():
    assert summarize_reward_overrides({"train": {"max_steps": 10}}, {}) == {}
    assert summarize_reward_overrides({}, {}) == {}


def test_summarize_reward_overrides_never_includes_a_pinned_key_in_practice():
    """Documents the invariant the two functions together maintain: by the
    time summarize_reward_overrides would run in main(), check_reward_overrides
    has already refused any pinned key, so main() never logs one as an
    ordinary override. This test only pins summarize_reward_overrides' own
    behaviour if it were ever called on a pinned key (it still records it --
    the refusal is main()'s job, not this function's).
    """
    overrides = {"reward": {"sigma_unknown": 0.0}}
    canonical_cfg = {"reward": {"sigma_unknown": 45.2}}
    record = summarize_reward_overrides(overrides, canonical_cfg)
    assert record == {"sigma_unknown": {"canonical": 45.2, "overridden_to": 0.0}}


def test_main_refuses_a_set_that_would_revert_ruling_f(capsys, monkeypatch, tmp_path):
    """Proves the exact scenario named in the review: an attempted
    ``--set reward.sigma_unknown=0.0`` must exit 1 before anything is loaded
    -- not even the config itself. ``load_config`` is replaced with a stub
    that raises if called at all, so this fails loudly (not just quietly
    returns the wrong thing) if the refusal gate is ever moved after it.
    """
    def _must_not_be_called(*_args, **_kwargs):
        raise AssertionError("main() proceeded past the refusal -- load_config was called")

    monkeypatch.setattr(train_grpo, "load_config", _must_not_be_called)

    exit_code = train_grpo.main(["--arm", "accuracy", "--set", "reward.sigma_unknown=0.0"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "sigma_unknown" in captured.err
    assert "Ruling F" in captured.err
    assert not (tmp_path / "results").exists()


def test_main_allows_a_set_on_an_unpinned_reward_key(monkeypatch, tmp_path):
    """The non-refused half of item 2: an unpinned ``reward.*`` override must
    still be allowed to proceed past the refusal gate, all the way to
    building the run (config load, run-dir creation, manifest logging). This
    stops the run right at ``load_frozen_baseline`` with a sentinel
    exception that main()'s own error handling does NOT catch (only
    ``FileNotFoundError``/``ValueError``/``KeyError`` are), so seeing it
    propagate proves execution reached that far -- i.e. the unpinned
    override was never refused.
    """
    sentinel = RuntimeError("stopped after load_frozen_baseline (by design)")

    def _raise(*_args, **_kwargs):
        raise sentinel

    monkeypatch.setattr(train_grpo, "load_frozen_baseline", _raise)

    with pytest.raises(RuntimeError, match="stopped after load_frozen_baseline"):
        train_grpo.main([
            "--arm", "accuracy", "--set", "reward.tolerance=50.0",
            "--set", f"output.results_root={tmp_path}",
        ])
