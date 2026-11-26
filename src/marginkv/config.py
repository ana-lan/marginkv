# src/marginkv/config.py
"""Load configs/default.yaml into a frozen, validated dataclass."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]  # src/marginkv/config.py -> repo root
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

VALID_DEVICES = {"auto", "mps", "cpu", "cuda"}
VALID_DTYPES = {"bfloat16", "float16", "float32"}


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    artifacts_dir: Path
    results_dir: Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    revision: str = "main"


@dataclass(frozen=True)
class Models:
    qwen3: ModelSpec
    bge: ModelSpec


@dataclass(frozen=True)
class Retrieval:
    method: str
    pool_size: int


@dataclass(frozen=True)
class Rerank:
    depth: int
    max_doc_tokens: int
    batch_size: int


@dataclass(frozen=True)
class Eval:
    cutoff: int
    recall_cutoff: int


@dataclass(frozen=True)
class Runtime:
    device: str
    dtype: str


@dataclass(frozen=True)
class Lora:
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    lr: float
    weight_decay: float
    warmup_frac: float
    epochs: int
    group_size: int
    groups_per_step: int
    micro_groups: int
    max_grad_norm: float
    temperature: float
    dev_every: int


@dataclass(frozen=True)
class Config:
    seed: int
    datasets: tuple[str, ...]
    split: str
    paths: Paths
    models: Models
    retrieval: Retrieval
    rerank: Rerank
    eval: Eval
    runtime: Runtime
    lora: Lora

    def to_dict(self) -> dict:
        # Round-trip through JSON so Paths become plain strings.
        return json.loads(json.dumps(asdict(self), default=str))


def _build(cls, data, where: str):
    """Construct a dataclass, failing loudly on missing or unknown keys."""
    if not isinstance(data, dict):
        raise ValueError(f"config section '{where}' must be a mapping, got {type(data).__name__}")
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown key(s) in '{where}': {sorted(unknown)}; allowed: {sorted(allowed)}")
    try:
        return cls(**data)
    except TypeError as e:
        raise ValueError(f"bad config section '{where}': {e}") from None


def _validate(cfg: Config) -> None:
    if cfg.rerank.depth > cfg.retrieval.pool_size:
        raise ValueError("rerank.depth must be <= retrieval.pool_size")
    if cfg.eval.cutoff > cfg.rerank.depth:
        raise ValueError("eval.cutoff must be <= rerank.depth")
    if cfg.eval.recall_cutoff > cfg.retrieval.pool_size:
        raise ValueError("eval.recall_cutoff must be <= retrieval.pool_size")
    if cfg.runtime.device not in VALID_DEVICES:
        raise ValueError(f"runtime.device must be one of {sorted(VALID_DEVICES)}")
    if cfg.runtime.dtype not in VALID_DTYPES:
        raise ValueError(f"runtime.dtype must be one of {sorted(VALID_DTYPES)}")
    if not cfg.datasets:
        raise ValueError("datasets must not be empty")
    L = cfg.lora
    if L.r <= 0 or L.group_size < 2 or L.groups_per_step <= 0 or L.micro_groups <= 0:
        raise ValueError("lora: r, group_size, groups_per_step, micro_groups must be positive (group_size >= 2)")
    if L.groups_per_step % L.micro_groups:
        raise ValueError("lora.groups_per_step must be divisible by lora.micro_groups")
    if not L.target_modules:
        raise ValueError("lora.target_modules must not be empty")


def load_config(path: str | Path | None = None) -> Config:
    """Load the config. Order: explicit path, then $MARGINKV_CONFIG, then configs/default.yaml."""
    path = Path(path or os.environ.get("MARGINKV_CONFIG", DEFAULT_CONFIG))
    raw = yaml.safe_load(path.read_text())

    top_allowed = {f.name for f in fields(Config)}
    unknown = set(raw) - top_allowed
    if unknown:
        raise ValueError(f"unknown top-level key(s) in {path.name}: {sorted(unknown)}")

    paths = _build(
        Paths,
        {k: (REPO_ROOT / v).resolve() for k, v in raw["paths"].items()},
        "paths",
    )
    models = _build(
        Models,
        {k: _build(ModelSpec, v, f"models.{k}") for k, v in raw["models"].items()},
        "models",
    )

    lora_raw = dict(raw["lora"])
    lora_raw["target_modules"] = tuple(lora_raw.get("target_modules", ()))
    for k in ("dropout", "lr", "weight_decay", "warmup_frac", "max_grad_norm", "temperature"):
        if k in lora_raw:
            lora_raw[k] = float(lora_raw[k])

    cfg = Config(
        seed=int(raw["seed"]),
        datasets=tuple(raw["datasets"]),
        split=str(raw["split"]),
        paths=paths,
        models=models,
        retrieval=_build(Retrieval, raw["retrieval"], "retrieval"),
        rerank=_build(Rerank, raw["rerank"], "rerank"),
        eval=_build(Eval, raw["eval"], "eval"),
        runtime=_build(Runtime, raw["runtime"], "runtime"),
        lora=_build(Lora, lora_raw, "lora"),
    )
    
    _validate(cfg)
    return cfg