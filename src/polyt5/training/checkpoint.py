"""Checkpoint save/load/resume/rotation.

Project rule: never save a model checkpoint without saving its configuration.
:func:`save_checkpoint` enforces this by refusing to write when
``model_config`` is missing.

Serialization note: we use ``torch.save`` / ``torch.load`` with
``weights_only=False`` because a checkpoint deliberately stores plain Python
objects alongside tensors — the resolved run config dict, the model config
dict, and the RNG snapshot (:class:`polyt5.utils.RngState`, which contains
numpy RNG state tuples). ``weights_only=True`` cannot represent those. The
security caveat (arbitrary-code unpickling) is acceptable because checkpoints
in this project are produced and consumed locally by our own code; the load
path guards against obviously wrong files by validating the payload shape.

This module is deliberately trainer-agnostic so a future RL trainer can reuse
it unchanged: it only needs a model, an optimizer, and plain metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from polyt5.utils import get_rng_state, set_rng_state
from polyt5.utils.seeding import RngState

_REQUIRED_KEYS = ("model_state", "model_config", "epoch", "global_step")


@dataclass
class Checkpoint:
    """Everything needed to resume training exactly where it stopped."""

    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    scheduler_state: dict[str, Any] | None
    epoch: int
    global_step: int
    config: dict[str, Any]
    model_config: dict[str, Any]
    tokenizer_path: str | None = None
    tokenizer_sha256: str | None = None
    rng_state: Any = None
    train_metrics: dict[str, Any] | None = None
    val_metrics: dict[str, Any] | None = None
    torch_version: str = field(default_factory=lambda: torch.__version__)
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    model_config: dict[str, Any],
    tokenizer_path: str | Path | None = None,
    tokenizer_sha256: str | None = None,
    train_metrics: dict[str, Any] | None = None,
    val_metrics: dict[str, Any] | None = None,
    rng_state: Any = None,
) -> Path:
    """Atomically write a checkpoint. Refuses to save without ``model_config``.

    Args:
        path: Destination ``.pt`` file.
        model: Model whose ``state_dict`` is saved.
        optimizer: Optional optimizer whose state is saved.
        scheduler: Optional LR scheduler whose state is saved.
        epoch: Index of the last COMPLETED epoch (0-based).
        global_step: Number of optimizer steps taken so far.
        config: The full resolved run config dict.
        model_config: ``PolyT5Config.to_dict()`` — required.
        tokenizer_path: Path of the tokenizer used (provenance).
        tokenizer_sha256: Hash of the tokenizer file (provenance).
        train_metrics: Latest training metrics.
        val_metrics: Latest validation metrics.
        rng_state: RNG snapshot; captured automatically when ``None``.

    Returns:
        The path written.

    Raises:
        ValueError: If ``model_config`` is falsy. Never save a model
            checkpoint without saving its configuration.
    """
    if not model_config:
        raise ValueError(
            "Refusing to save a checkpoint without model_config: a checkpoint "
            "that cannot rebuild its model is unusable. Pass "
            "model_config=PolyT5Config.to_dict()."
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = Checkpoint(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict() if optimizer is not None else None,
        scheduler_state=scheduler.state_dict() if scheduler is not None else None,
        epoch=int(epoch),
        global_step=int(global_step),
        config=dict(config) if config else {},
        model_config=dict(model_config),
        tokenizer_path=str(tokenizer_path) if tokenizer_path is not None else None,
        tokenizer_sha256=tokenizer_sha256,
        rng_state=rng_state if rng_state is not None else get_rng_state(),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )

    # NOT dataclasses.asdict(): that would deep-copy every tensor in
    # model_state before pickling. A shallow field dict is enough.
    payload = {f.name: getattr(checkpoint, f.name) for f in fields(checkpoint)}

    # Atomic write: never leave a truncated checkpoint behind on crash.
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint dict written by :func:`save_checkpoint`.

    ``weights_only=False`` is required because checkpoints store Python
    objects (config dicts, numpy RNG state); see the module docstring. The
    payload shape is validated as a guard against loading arbitrary files.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a polyT5 checkpoint (got {type(payload).__name__})")
    missing = [k for k in _REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(f"{path} is missing checkpoint keys: {missing}")
    return payload


def resume(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    restore_rng: bool = True,
) -> tuple[int, int]:
    """Restore model/optimizer/scheduler/RNG state in place.

    Returns:
        ``(epoch, global_step)`` — the last completed epoch and the number of
        optimizer steps at save time.
    """
    payload = load_checkpoint(path)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if restore_rng and payload.get("rng_state") is not None:
        state = payload["rng_state"]
        if isinstance(state, dict):  # tolerate asdict()-flattened snapshots
            state = RngState(**state)
        set_rng_state(state)
    return int(payload["epoch"]), int(payload["global_step"])


def rotate_checkpoints(
    directory: str | Path,
    keep_last: int,
    keep_best: str | None = None,
) -> list[Path]:
    """Delete old ``*.pt`` checkpoints, keeping the newest ``keep_last``.

    Args:
        directory: Directory containing checkpoint files.
        keep_last: Number of most-recent checkpoints to keep (by mtime,
            tie-broken by filename so ``epoch_0004`` outranks ``epoch_0003``).
        keep_best: Optional metric name (looked up in each checkpoint's
            ``val_metrics``, LOWER is better, e.g. ``"val_loss"``); the best
            checkpoint by that metric is kept in addition to the newest.

    The file named ``best.pt`` is NEVER deleted regardless of age.

    Returns:
        The checkpoint paths remaining on disk, oldest first.
    """
    directory = Path(directory)
    candidates = sorted(
        (p for p in directory.glob("*.pt") if p.name != "best.pt"),
        key=lambda p: (p.stat().st_mtime, p.name),
    )
    keep: set[Path] = set(candidates[-keep_last:]) if keep_last > 0 else set()

    if keep_best is not None:
        best_path: Path | None = None
        best_value = float("inf")
        for candidate in candidates:
            try:
                payload = torch.load(candidate, map_location="cpu", weights_only=False)
                value = (payload.get("val_metrics") or {}).get(keep_best)
            except Exception:
                continue
            if value is not None and value < best_value:
                best_value, best_path = value, candidate
        if best_path is not None:
            keep.add(best_path)

    for candidate in candidates:
        if candidate not in keep:
            candidate.unlink()

    remaining = [p for p in candidates if p in keep]
    pinned = directory / "best.pt"
    if pinned.exists():
        remaining.append(pinned)
    return remaining
