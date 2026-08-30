"""Tests for the round-2 kl_coef sweep driver."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_kl_sweep  # noqa: E402


def test_experiment_name_reads_the_configs_own_key(tmp_path):
    """The driver must write where train_grpo.py writes, or resume polls a dead path."""
    cfg = tmp_path / "composite_kl05.yaml"
    cfg.write_text("experiment_name: grpo_composite_kl05\ntrain:\n  kl_coef: 0.05\n",
                   encoding="utf-8")
    assert run_kl_sweep.experiment_name(cfg) == "grpo_composite_kl05"


def test_experiment_name_falls_back_to_the_config_stem(tmp_path):
    cfg = tmp_path / "composite_kl9.yaml"
    cfg.write_text("train:\n  kl_coef: 0.9\n", encoding="utf-8")
    assert run_kl_sweep.experiment_name(cfg) == "grpo_composite_kl9"


def test_the_shipped_sweep_configs_differ_only_in_kl_coef():
    """One variable, or the sweep answers nothing.

    A second differing key would confound every conclusion drawn from the sweep, and
    the difference would be invisible in the results.
    """
    import yaml

    base = yaml.safe_load((REPO_ROOT / "configs" / "rl" / "composite.yaml").read_text(
        encoding="utf-8"))
    for config in run_kl_sweep.DEFAULT_CONFIGS:
        loaded = yaml.safe_load(config.read_text(encoding="utf-8"))
        differing = {k for k in set(base) | set(loaded) if base.get(k) != loaded.get(k)}
        assert differing == {"experiment_name", "train"}, f"{config.name}: {differing}"
        train_diff = {
            k for k in set(base["train"]) | set(loaded["train"])
            if base["train"].get(k) != loaded["train"].get(k)
        }
        assert train_diff == {"kl_coef"}, f"{config.name}: train differs in {train_diff}"


def test_every_sweep_config_raises_kl_above_round_ones_value():
    import yaml

    base = yaml.safe_load((REPO_ROOT / "configs" / "rl" / "composite.yaml").read_text(
        encoding="utf-8"))["train"]["kl_coef"]
    values = [
        yaml.safe_load(c.read_text(encoding="utf-8"))["train"]["kl_coef"]
        for c in run_kl_sweep.DEFAULT_CONFIGS
    ]
    assert all(v > base for v in values), f"{values} vs round-1 {base}"
    assert values == sorted(values), "configs should sweep upward for a readable frontier"


def test_resume_target_skips_a_finished_run(tmp_path):
    run = tmp_path / "grpo_composite_kl05"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "step_002000.pt").write_bytes(b"x")
    (run / "metrics.csv").write_text("step,reward_mean\n2000,0.9\n", encoding="utf-8")
    assert run_kl_sweep.resume_target(run, 2000) is None


def test_resume_target_points_at_a_partial_run(tmp_path):
    run = tmp_path / "grpo_composite_kl05"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "step_000300.pt").write_bytes(b"x")
    (run / "metrics.csv").write_text("step,reward_mean\n300,0.7\n", encoding="utf-8")
    assert run_kl_sweep.resume_target(run, 2000) == run / "checkpoints"


def test_a_missing_config_aborts_before_spawning_anything(tmp_path, capsys, monkeypatch):
    def _must_not_run(*_a, **_kw):
        raise AssertionError("train() must not be reached for a missing config")

    monkeypatch.setattr(run_kl_sweep, "train", _must_not_run)
    assert run_kl_sweep.main(["--configs", str(tmp_path / "nope.yaml")]) == 1
    assert "no such config" in capsys.readouterr().out


def test_the_chain_stops_at_the_first_failure(monkeypatch, tmp_path):
    """A dead run must not hand a broken GPU state to the next config."""
    calls = []

    def fake_train(config, **_kw):
        calls.append(config.name)
        return config.name != "b.yaml"

    for name in ("a.yaml", "b.yaml", "c.yaml"):
        (tmp_path / name).write_text("train: {kl_coef: 0.1}\n", encoding="utf-8")
    monkeypatch.setattr(run_kl_sweep, "train", fake_train)
    paths = [str(tmp_path / n) for n in ("a.yaml", "b.yaml", "c.yaml")]
    rc = run_kl_sweep.main(["--configs", *paths])
    assert rc == 1
    assert calls == ["a.yaml", "b.yaml"], "c.yaml must never start"
