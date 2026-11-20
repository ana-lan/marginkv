# src/marginkv/utils/seed.py
"""Seed every RNG we use."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds MPS
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # warn_only: some MPS ops have no deterministic version; warn instead of crashing.
        torch.use_deterministic_algorithms(True, warn_only=True)
    # Note: PYTHONHASHSEED can't be changed for the running process.
    # Never rely on set/dict-of-set ordering for doc IDs; always sort.