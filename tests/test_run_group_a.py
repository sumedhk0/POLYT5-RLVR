# tests/test_run_group_a.py
"""Group A Task 13: the runner's guards. Nothing here trains anything."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from polyt5.training.group_a import ARM_IDS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_group_a  # noqa: E402


def write_splits(path: Path, *, n: int = 10, n_splits: int = 2) -> Path:
    splits = []
    for index in range(n_splits):
        order = [(i + index) % n for i in range(n)]
        splits.append(
            {"index": index, "train": order[:6], "val": order[6:8], "test": order[8:]}
        )
    path.write_text(
        json.dumps({"task": "tg_prediction", "n": n, "base_seed": 0,
                    "n_splits": n_splits, "splits": splits}),
        encoding="utf-8",
    )
    return path


def test_frozen_splits_load_and_validate(tmp_path):
    path = write_splits(tmp_path / "splits.json")
    splits = run_group_a.load_frozen_splits(path, n_examples=10)
    assert [s.index for s in splits] == [0, 1]
    assert len(splits[0].train) == 6
    assert len(splits[0].test) == 2


def test_a_corpus_of_the_wrong_size_is_refused(tmp_path):
    """The one guard that keeps every Group A number comparable to 28.67 K."""
    path = write_splits(tmp_path / "splits.json", n=10)
    with pytest.raises(ValueError, match="7354|9|indices"):
        run_group_a.load_frozen_splits(path, n_examples=9)


def test_overlapping_train_and_test_is_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"n": 4, "n_splits": 1,
                    "splits": [{"index": 0, "train": [0, 1, 2], "val": [], "test": [2, 3]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="disjoint"):
        run_group_a.load_frozen_splits(path, n_examples=4)


def test_a_split_that_does_not_cover_every_index_is_refused(tmp_path):
    path = tmp_path / "short.json"
    path.write_text(
        json.dumps({"n": 4, "n_splits": 1,
                    "splits": [{"index": 0, "train": [0, 1], "val": [], "test": [2]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cover"):
        run_group_a.load_frozen_splits(path, n_examples=4)


def test_a_file_with_no_splits_is_refused(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"n": 4, "splits": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no splits"):
        run_group_a.load_frozen_splits(path, n_examples=4)


def test_resolve_arms_defaults_to_all_seven_in_order():
    assert [c.arm for c in run_group_a.resolve_arms(None)] == list(ARM_IDS)


def test_resolve_arms_honours_a_subset_and_deduplicates():
    assert [c.arm for c in run_group_a.resolve_arms(["A3", "B0", "A3"])] == ["B0", "A3"]


def test_resolve_arms_passes_hyperparameter_overrides_through():
    configs = run_group_a.resolve_arms(["A2", "A3"], descriptor_lambda=0.4, n_writings=6)
    assert configs[0].descriptor_lambda == 0.4
    assert configs[1].n_writings == 6


def test_resolve_arms_rejects_an_unknown_arm():
    with pytest.raises(ValueError, match="A9"):
        run_group_a.resolve_arms(["A9"])


def test_baseline_reference_comes_from_the_frozen_artifact(tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text(
        json.dumps({"tg_prediction_5split": {"mae": {"mean": 28.6733, "std": 0.7591}}}),
        encoding="utf-8",
    )
    assert run_group_a.load_baseline_reference(path) == (28.6733, 0.7591)


def test_baseline_reference_refuses_an_artifact_without_the_key(tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps({"tg_prediction_5split": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="mae"):
        run_group_a.load_baseline_reference(path)


def test_the_real_frozen_artifact_still_says_what_the_plan_assumes():
    repo = Path(__file__).resolve().parents[1]
    artifact = repo / "artifacts" / "baseline" / "frozen_baseline.json"
    if not artifact.is_file():
        pytest.skip("frozen baseline artifact missing")
    mean, std = run_group_a.load_baseline_reference(artifact)
    assert mean == pytest.approx(28.6733, abs=1e-4)
    assert std == pytest.approx(0.7591, abs=1e-4)


def test_default_output_root_is_not_a_live_run_directory():
    args = run_group_a.parse_args([])
    assert Path(args.out).as_posix().endswith("results/group_a")
    assert "grpo" not in Path(args.out).as_posix()


def test_arms_can_be_selected_on_the_command_line():
    args = run_group_a.parse_args(["--arm", "A1", "--arm", "A4"])
    assert args.arm == ["A1", "A4"]
