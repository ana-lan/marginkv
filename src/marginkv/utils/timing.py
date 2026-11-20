# src/marginkv/utils/timing.py
"""Wall-clock timers that sync the device first, plus latency summaries."""
from __future__ import annotations

import time

import numpy as np
import torch

from marginkv.utils.device import sync


class Timer:
    """with Timer(device) as t: ...  then read t.ms"""

    def __init__(self, device: torch.device | str | None = None):
        self.device = device
        self.ms: float | None = None

    def __enter__(self):
        sync(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync(self.device)
        self.ms = (time.perf_counter() - self._t0) * 1000.0
        return False


def summarize_ms(samples: list[float]) -> dict:
    if not samples:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }