"""Unit coverage for ``scripts/run_round1.py``'s multi-seed chaining (Addition 2).

None of these need a real training run: ``subprocess.run`` is stubbed out
everywhere, so these only pin the COMMAND and directory-naming logic, not
anything that actually trains.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_round1  # noqa: E402

# ------------------------------------------------------------------ run_dir_for


def test_run_dir_for_seed_zero_is_unsuffixed():
    """Backward compatibility: the in-flight results/grpo_accuracy/ run."""
    assert run_round1.run_dir_for("accuracy") == REPO_ROOT / "results" / "grpo_accuracy"
    assert run_round1.run_dir_for("accuracy", seed=0) == REPO_ROOT / "results" / "grpo_accuracy"


def test_run_dir_for_nonzero_seed_gets_a_suffix():
    assert (run_round1.run_dir_for("accuracy", seed=1)
            == REPO_ROOT / "results" / "grpo_accuracy_seed1")
    assert (run_round1.run_dir_for("composite", seed=3)
            == REPO_ROOT / "results" / "grpo_composite_seed3")


def test_run_dir_for_matches_train_grpo_run_experiment_name():
    """The two must never drift apart -- this pins them together directly,
    against the base name each REAL configs/rl/<arm>.yaml actually declares
    (not a re-hardcoded f"grpo_{arm}" on both sides, which cannot see
    base-name drift -- see review finding 3 and
    test_run_dir_for_follows_a_configs_own_experiment_name_override below).
    """
    import train_grpo

    for arm, seed in (("accuracy", 0), ("accuracy", 1), ("composite", 5)):
        base_name = run_round1._experiment_name_for(arm)
        expected = REPO_ROOT / "results" / train_grpo.run_experiment_name(base_name, seed)
        assert run_round1.run_dir_for(arm, seed=seed) == expected


def test_experiment_name_for_reads_the_real_yaml_files():
    """Every shipped config's own experiment_name key, read the same way
    train_grpo.main() reads it -- not assumed to equal f"grpo_{arm}".
    """
    for arm in ("accuracy", "validity", "composite", "constraint", "control"):
        raw = yaml.safe_load(
            (REPO_ROOT / "configs" / "rl" / f"{arm}.yaml").read_text(encoding="utf-8"))
        expected = raw.get("experiment_name") or f"grpo_{arm}"
        assert run_round1._experiment_name_for(arm) == expected


def test_run_dir_for_follows_a_configs_own_experiment_name_override(tmp_path, monkeypatch):
    """Review finding 3, reproduced directly: the pin test used to hardcode
    f"grpo_{arm}" on BOTH sides, so it could not see base-name drift. Here
    a YAML genuinely overrides experiment_name, and the driver must follow
    it -- not silently keep polling the f"grpo_{arm}" directory nobody is
    writing to.
    """
    monkeypatch.setattr(run_round1, "REPO", tmp_path)
    (tmp_path / "configs" / "rl").mkdir(parents=True)
    (tmp_path / "configs" / "rl" / "composite.yaml").write_text(
        "experiment_name: grpo_composite_v2\narm: composite\n", encoding="utf-8")

    assert (run_round1.run_dir_for("composite", seed=0)
            == tmp_path / "results" / "grpo_composite_v2")
    assert (run_round1.run_dir_for("composite", seed=1)
            == tmp_path / "results" / "grpo_composite_v2_seed1")
    assert not (tmp_path / "results" / "grpo_composite").exists()


def test_experiment_name_for_falls_back_when_the_config_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(run_round1, "REPO", tmp_path)
    (tmp_path / "configs" / "rl").mkdir(parents=True)
    assert run_round1._experiment_name_for("accuracy") == "grpo_accuracy"


# --------------------------------------------------------------------- last_step


def test_last_step_reads_the_seeded_run_directorys_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(run_round1, "REPO", tmp_path)
    run_dir = tmp_path / "results" / "grpo_accuracy_seed2"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.csv").write_text("step,loss\n10,0.5\n20,0.4\n", encoding="utf-8")
    assert run_round1.last_step("accuracy", seed=2) == 20


def test_last_step_none_when_metrics_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(run_round1, "REPO", tmp_path)
    assert run_round1.last_step("accuracy", seed=0) is None


# ------------------------------------------------------------------------- train


def test_train_seed_zero_calls_train_grpo_without_a_set_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(run_round1, "REPO", tmp_path)
    captured = {}

    class _Completed:
        returncode = 0

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _Completed()

    monkeypatch.setattr(run_round1.subprocess, "run", fake_run)

    assert run_round1.train("accuracy", 0) is True
    assert captured["cmd"] == [
        run_round1.PYTHON, "-u", str(tmp_path / "scripts" / "train_grpo.py"),
        "--arm", "accuracy",
    ]
    assert (tmp_path / "results" / "rl_accuracy_round1.log").exists()


def test_train_nonzero_seed_appends_a_set_seed_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(run_round1, "REPO", tmp_path)
    captured = {}

    class _Completed:
        returncode = 0

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _Completed()

    monkeypatch.setattr(run_round1.subprocess, "run", fake_run)

    assert run_round1.train("accuracy", 3) is True
    assert captured["cmd"] == [
        run_round1.PYTHON, "-u", str(tmp_path / "scripts" / "train_grpo.py"),
        "--arm", "accuracy", "--set", "seed=3",
    ]
    assert (tmp_path / "results" / "rl_accuracy_seed3_round1.log").exists()


def test_train_returns_false_on_nonzero_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(run_round1, "REPO", tmp_path)

    class _Completed:
        returncode = 1

    monkeypatch.setattr(run_round1.subprocess, "run", lambda cmd, **kwargs: _Completed())

    assert run_round1.train("accuracy", 0) is False


# --------------------------------------------------------------------------- main


def test_main_defaults_to_seed_zero_only_matching_pre_multiseed_behaviour(monkeypatch):
    calls = []
    monkeypatch.setattr(run_round1, "train", lambda arm, seed=0: calls.append((arm, seed)) or True)
    monkeypatch.setattr(sys, "argv", ["run_round1.py", "--then", "validity", "composite"])

    assert run_round1.main() == 0
    assert calls == [("validity", 0), ("composite", 0)]


def test_main_chains_seed_outer_arm_inner(monkeypatch):
    """--seeds 0 1 --then a b runs (a,0) (b,0) (a,1) (b,1), in that order --
    every arm finishes a seed before the next seed starts.
    """
    calls = []
    monkeypatch.setattr(run_round1, "train", lambda arm, seed=0: calls.append((arm, seed)) or True)
    monkeypatch.setattr(
        sys, "argv",
        ["run_round1.py", "--then", "accuracy", "composite", "--seeds", "0", "1"],
    )

    assert run_round1.main() == 0
    assert calls == [
        ("accuracy", 0), ("composite", 0), ("accuracy", 1), ("composite", 1),
    ]


def test_main_stops_the_whole_chain_on_a_failed_arm(monkeypatch):
    calls = []

    def fake_train(arm, seed=0):
        calls.append((arm, seed))
        return arm != "composite"

    monkeypatch.setattr(run_round1, "train", fake_train)
    monkeypatch.setattr(
        sys, "argv",
        ["run_round1.py", "--then", "accuracy", "composite", "constraint", "--seeds", "0", "1"],
    )

    assert run_round1.main() == 1
    assert calls == [("accuracy", 0), ("composite", 0)]
