"""Training system: optimizer/scheduler builders, checkpointing, the Trainer.

Plain PyTorch only — no HuggingFace Trainer, no Lightning, no accelerate.

``GroupATrainer`` and ``InterleavedLoader`` are imported LAZILY (see
``__getattr__`` below), not at module load. ``import polyt5.training`` is on
the path of every existing supervised entry point, including the
92.3M-sequence pretraining run, none of which need Group A's modules;
``polyt5.training.multitask_trainer`` in turn imports ``polyt5.data.multitask``
and ``polyt5.model.multitask``, so an eager import here would mean an
import-time failure in either new module breaks the frozen-baseline training
path, which previously depended on neither. ``from polyt5.training import
GroupATrainer`` still works — the import just happens on first attribute
access instead of at package-import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from polyt5.training.checkpoint import (
    Checkpoint,
    load_checkpoint,
    resume,
    rotate_checkpoints,
    save_checkpoint,
)
from polyt5.training.cycle import CycleConfig, build_cycle_loss
from polyt5.training.group_a import ARM_IDS, SWITCH_NAMES, GroupAConfig, arm_config
from polyt5.training.optim import build_optimizer, build_scheduler
from polyt5.training.trainer import Trainer, TrainerConfig

if TYPE_CHECKING:  # pragma: no cover - typing only; see the lazy __getattr__ below
    from polyt5.training.multitask_trainer import GroupATrainer, InterleavedLoader

__all__ = [
    "ARM_IDS",
    "Checkpoint",
    "CycleConfig",
    "GroupAConfig",
    "GroupATrainer",
    "InterleavedLoader",
    "SWITCH_NAMES",
    "Trainer",
    "TrainerConfig",
    "arm_config",
    "build_cycle_loss",
    "build_optimizer",
    "build_scheduler",
    "load_checkpoint",
    "resume",
    "rotate_checkpoints",
    "save_checkpoint",
]

_LAZY = {"GroupATrainer", "InterleavedLoader"}


def __getattr__(name: str) -> Any:
    """Import ``multitask_trainer`` only when one of its symbols is touched."""
    if name in _LAZY:
        from polyt5.training import multitask_trainer

        return getattr(multitask_trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
