# src/marginkv/utils/io.py
"""Save/load helpers. Every write is atomic: write to .tmp, then rename."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from marginkv.config import REPO_ROOT

_TRACKED_PACKAGES = ["torch", "transformers", "tokenizers", "safetensors", "huggingface_hub",
                     "bm25s", "PyStemmer", "pytrec-eval-terrier", "numpy", "pandas"]

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(o):
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (np.ndarray, torch.Tensor)):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _atomic_target(path: Path) -> Path:
    ensure_dir(path.parent)
    return path.with_name(path.name + ".tmp")


def save_json(obj, path: str | Path, indent: int = 2) -> Path:
    path = Path(path)
    tmp = _atomic_target(path)
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, default=_json_default)
        f.write("\n")
    os.replace(tmp, path)
    return path


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def save_pt(obj, path: str | Path) -> Path:
    path = Path(path)
    tmp = _atomic_target(path)
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return path


def load_pt(path: str | Path, weights_only: bool = True):
    return torch.load(path, map_location="cpu", weights_only=weights_only)


def save_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    tmp = _atomic_target(path)
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run_info() -> dict:
    """Provenance block to embed in every results file."""
    pkgs = {}
    for name in _TRACKED_PACKAGES:
        try:
            pkgs[name] = version(name)
        except PackageNotFoundError:
            pkgs[name] = None
    status = _git("status", "--porcelain")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
        "packages": pkgs,
    }