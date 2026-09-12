"""Device selection. Kept deliberately thin: one place that knows about hardware."""

from __future__ import annotations

import torch


def get_device(spec: str | torch.device = "auto") -> torch.device:
    if isinstance(spec, torch.device):
        return spec
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def supports_triton(device: torch.device) -> bool:
    """Phase 2 gate: the paged-attention kernel only runs on CUDA."""
    return device.type == "cuda"


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        major, minor = torch.cuda.get_device_capability(device)
        return f"{name} (sm_{major}{minor})"
    return device.type
