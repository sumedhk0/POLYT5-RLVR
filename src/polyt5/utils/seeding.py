"""Deterministic seeding utilities.

Every experiment in this repository must be reproducible from (config, seed).
Capture/restore helpers exist so that a checkpoint can resume the *exact* RNG
state, which matters for span corruption (a stochastic data transform) and for
sampling-based generation.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RngState:
    """Snapshot of every RNG stream this project uses."""

    python: Any
    numpy: Any
    torch: Any | None = None
    torch_cuda: Any | None = None


def seed_everything(seed: int, *, deterministic: bool = False) -> int:
    """Seed python, numpy and torch (if importable).

    Args:
        seed: Base seed.
        deterministic: If True, ask cuDNN/torch for deterministic kernels. This
            is slower and is intended for unit tests and debugging, not for
            large-scale pretraining.

    Returns:
        The seed that was applied (echoed for logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # chemistry-only environments must still work
        pass
    return seed


def get_rng_state() -> RngState:
    """Capture RNG state for checkpointing."""
    torch_state = None
    cuda_state = None
    try:
        import torch

        torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return RngState(python=random.getstate(), numpy=np.random.get_state(), torch=torch_state,
                    torch_cuda=cuda_state)


def set_rng_state(state: RngState) -> None:
    """Restore RNG state captured by :func:`get_rng_state`."""
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    if state.torch is not None:
        try:
            import torch

            torch.set_rng_state(state.torch)
            if state.torch_cuda is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state.torch_cuda)
        except ImportError:
            pass
