# src/marginkv/utils/device.py
"""Device and dtype selection, plus a device-aware sync."""
from __future__ import annotations

import torch

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def get_device(pref: str = "auto") -> torch.device:
    """auto picks cuda, then mps, then cpu. An explicit choice that isn't available raises."""
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    if pref == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device=mps requested but MPS is not available (Rosetta Python?)")
    return torch.device(pref)


def get_dtype(name: str) -> torch.dtype:
    if name not in _DTYPES:
        raise ValueError(f"unknown dtype '{name}', expected one of {sorted(_DTYPES)}")
    return _DTYPES[name]


def sync(device: torch.device | str | None) -> None:
    """Block until queued GPU work finishes. Required before reading any timer."""
    if device is None:
        return
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()