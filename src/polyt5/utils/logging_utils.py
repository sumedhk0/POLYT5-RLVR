"""Console logging, machine-readable metric logging, and the on-disk run layout.

Layout produced by :class:`RunDirectory` (see ``docs/reproduction.md``)::

    results/<experiment_name>/
        config.yaml          fully-resolved config for this run
        manifest.json        environment, device, git commit, tokenizer hash
        metrics.jsonl        one JSON object per logged step/epoch
        metrics.csv          the same rows, flattened
        predictions.jsonl    downstream property-prediction outputs
        generations.jsonl    sampled polymers
        logs/                console log files
        checkpoints/         model + optimizer state
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def get_logger(
    name: str,
    *,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Return a console logger, optionally tee'd to a file. Idempotent per name."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S"
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def _git_commit() -> str | None:
    """Current commit hash, or None outside a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except Exception:
        return None


@dataclass
class RunDirectory:
    """Owns the on-disk artifacts of a single experiment."""

    root: Path

    @classmethod
    def create(cls, results_root: str | Path, experiment_name: str) -> RunDirectory:
        root = Path(results_root) / experiment_name
        for sub in ("", "logs", "checkpoints"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        return cls(root=root)

    # ------------------------------------------------------------------ paths
    @property
    def config_path(self) -> Path:
        return self.root / "config.yaml"

    @property
    def metrics_jsonl(self) -> Path:
        return self.root / "metrics.jsonl"

    @property
    def metrics_csv(self) -> Path:
        return self.root / "metrics.csv"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    # ---------------------------------------------------------------- writers
    def write_manifest(self, extra: dict[str, Any] | None = None) -> Path:
        """Record everything needed to explain *how* this run was produced."""
        manifest: dict[str, Any] = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "git_commit": _git_commit(),
            "argv": sys.argv,
        }
        try:
            import torch

            manifest["torch"] = torch.__version__
            manifest["cuda"] = torch.version.cuda
        except ImportError:
            pass
        if extra:
            manifest.update(extra)
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        return path

    def archive_previous_metrics(self) -> list[Path]:
        """Move any existing metric files aside so a FRESH run starts clean.

        :meth:`log_metrics` appends, which is right for a resumed run and wrong for a
        restarted one: the restarted run's rows land after the dead run's, and the file
        then contains two runs interleaved with duplicate step numbers.

        That is not merely untidy. ``scripts/run_round1.py`` decides whether an arm has
        finished by taking ``max(step)`` from this file, so a dead run that reached step
        2000 would make the driver treat a freshly started arm as already complete and
        skip straight past it. A stale row is indistinguishable from a live one.

        Renames rather than deletes -- an interrupted run's metrics are often the only
        record of why it died.

        Returns:
            The archive paths written, empty if there was nothing to move.
        """
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        moved: list[Path] = []
        for path in (self.metrics_csv, self.metrics_jsonl):
            if not path.exists():
                continue
            target = path.with_name(f"{path.stem}.{stamp}{path.suffix}")
            n = 1
            while target.exists():
                target = path.with_name(f"{path.stem}.{stamp}.{n}{path.suffix}")
                n += 1
            path.rename(target)
            moved.append(target)
        return moved

    def log_metrics(self, record: dict[str, Any]) -> None:
        """Append one metric row to both ``metrics.jsonl`` and ``metrics.csv``."""
        record = {"wall_utc": datetime.now(timezone.utc).isoformat(), **record}
        with self.metrics_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

        write_header = not self.metrics_csv.exists()
        existing_fields: list[str] = []
        if not write_header:
            with self.metrics_csv.open("r", encoding="utf-8", newline="") as fh:
                existing_fields = next(csv.reader(fh), [])
        fields = list(existing_fields) or list(record.keys())
        for key in record:
            if key not in fields:
                fields.append(key)
        if not write_header and fields != existing_fields:
            # A new metric key appeared: rewrite the file with the wider header.
            with self.metrics_csv.open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            with self.metrics_csv.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        with self.metrics_csv.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow({k: record.get(k, "") for k in fields})

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def append_jsonl(self, name: str, rows: Any) -> Path:
        path = self.root / name
        rows = rows if isinstance(rows, list) else [rows]
        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=str) + "\n")
        return path
