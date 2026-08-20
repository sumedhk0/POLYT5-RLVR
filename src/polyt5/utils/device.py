"""Device selection and VRAM reporting.

The published polyT5 training used a single NVIDIA L40S (48 GB). This
reproduction targets a 12 GB RTX 4080 Laptop GPU, so every training entry point
must report memory and support gradient accumulation instead of assuming the
paper's physical batch size fits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DeviceInfo:
    device: str
    name: str
    total_memory_gb: float | None
    capability: str | None
    bf16_supported: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_device(prefer: str = "auto") -> str:
    """Return a torch device string.

    Args:
        prefer: ``"auto"``, ``"cuda"``, or ``"cpu"``.
    """
    import torch

    if prefer == "cpu":
        return "cpu"
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    if prefer == "cuda":
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return "cpu"


def describe_device(device: str | None = None) -> DeviceInfo:
    """Collect human- and machine-readable device facts for the run manifest."""
    import torch

    device = device or select_device()
    if device.startswith("cuda"):
        idx = 0 if ":" not in device else int(device.split(":")[1])
        props = torch.cuda.get_device_properties(idx)
        return DeviceInfo(
            device=device,
            name=props.name,
            total_memory_gb=round(props.total_memory / 1024**3, 2),
            capability=f"{props.major}.{props.minor}",
            bf16_supported=torch.cuda.is_bf16_supported(),
        )
    return DeviceInfo(device="cpu", name="cpu", total_memory_gb=None, capability=None,
                      bf16_supported=False)


def memory_stats(device: str = "cuda") -> dict[str, float]:
    """Peak/current allocator statistics in GB (empty dict on CPU)."""
    import torch

    if not device.startswith("cuda") or not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
    }


def reset_memory_stats(device: str = "cuda") -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
