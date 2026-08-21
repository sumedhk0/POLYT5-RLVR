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

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_grpo import (  # noqa: E402
    _latest_checkpoint,
    build_reward_arm,
    build_reward_ensemble,
    load_frozen_baseline,
    sha256_of_file,
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


def test_build_drift_monitor_refuses_an_auditor_that_leaks_into_the_ensemble(tmp_path):
    """The third independent guard on the same invariant, placed at the only
    point in the TRAINING path that opens the auditor's checkpoint at all.
    """
    from train_grpo import build_drift_monitor

    frozen = {
        "artifacts": {"split4": {"path": "nope.pt", "sha256": "0" * 64}},
        "reward_ensemble": ["split4"],
        "auditor": "split4",
    }
    with pytest.raises(ValueError, match="auditor"):
        build_drift_monitor(frozen, {}, tmp_path / "tok.json", device="cpu",
                            batch_size=1, num_beams=1, seed=0)


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
