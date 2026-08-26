"""A fresh run must not append to a dead run's metrics."""

from __future__ import annotations

import csv

from polyt5.utils.logging_utils import RunDirectory


def _steps(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return [int(r["step"]) for r in csv.DictReader(fh)]


def test_a_restarted_run_does_not_inherit_the_dead_runs_steps(tmp_path):
    """The bug: killed run reached step 20, restart logs step 10, file reads [10, 20, 10].

    run_round1.py takes max(step) to decide an arm finished, so a dead run that had
    reached 2000 would make a freshly started arm look complete and be skipped.
    """
    run = RunDirectory.create(tmp_path, "grpo_novelty")
    run.log_metrics({"step": 10, "reward_mean": 0.5})
    run.log_metrics({"step": 20, "reward_mean": 0.6})

    run.archive_previous_metrics()
    run.log_metrics({"step": 10, "reward_mean": 0.9})

    assert _steps(run.metrics_csv) == [10]
    assert max(_steps(run.metrics_csv)) == 10


def test_the_dead_runs_metrics_are_preserved_not_deleted(tmp_path):
    """An interrupted run's metrics are often the only record of why it died."""
    run = RunDirectory.create(tmp_path, "grpo_novelty")
    run.log_metrics({"step": 10, "reward_mean": 0.5})
    run.log_metrics({"step": 20, "reward_mean": 0.6})

    moved = run.archive_previous_metrics()

    assert len(moved) == 2
    archived_csv = next(p for p in moved if p.suffix == ".csv")
    assert archived_csv.exists()
    assert _steps(archived_csv) == [10, 20]


def test_archiving_an_empty_run_directory_is_a_no_op(tmp_path):
    run = RunDirectory.create(tmp_path, "grpo_novelty")
    assert run.archive_previous_metrics() == []
    assert not run.metrics_csv.exists()


def test_two_archives_in_the_same_second_do_not_collide(tmp_path):
    """The timestamp has one-second resolution; a retry loop can restart faster."""
    run = RunDirectory.create(tmp_path, "grpo_novelty")
    run.log_metrics({"step": 10})
    first = run.archive_previous_metrics()
    run.log_metrics({"step": 10})
    second = run.archive_previous_metrics()

    assert first and second
    assert {p.name for p in first}.isdisjoint({p.name for p in second})
    assert all(p.exists() for p in first + second)


def test_resuming_still_appends(tmp_path):
    """Archiving is opt-in: a resumed run must keep its history in one file."""
    run = RunDirectory.create(tmp_path, "grpo_novelty")
    run.log_metrics({"step": 10})
    run.log_metrics({"step": 20})
    run.log_metrics({"step": 30})
    assert _steps(run.metrics_csv) == [10, 20, 30]
