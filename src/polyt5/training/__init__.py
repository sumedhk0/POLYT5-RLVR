"""Training system: optimizer/scheduler builders, checkpointing, the Trainer.

Plain PyTorch only — no HuggingFace Trainer, no Lightning, no accelerate.
"""

from polyt5.training.checkpoint import (
    Checkpoint,
    load_checkpoint,
    resume,
    rotate_checkpoints,
    save_checkpoint,
)
from polyt5.training.cycle import CycleConfig, build_cycle_loss
from polyt5.training.group_a import ARM_IDS, SWITCH_NAMES, GroupAConfig, arm_config
from polyt5.training.multitask_trainer import GroupATrainer, InterleavedLoader
from polyt5.training.optim import build_optimizer, build_scheduler
from polyt5.training.trainer import Trainer, TrainerConfig

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
