# scripts/00_download_models.py
"""Resolve each model's current commit hash, download exactly that snapshot, record it.

After running, paste the printed hashes into configs/default.yaml (models.*.revision).
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from marginkv.config import load_config
from marginkv.utils.io import run_info, save_json

# Weights in safetensors only, plus configs and tokenizer files. Skips .bin, onnx, etc.
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"]


def _dir_size_bytes(path: Path) -> int:
    # Snapshot files are symlinks into blobs/; stat() follows them to the real size.
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    cfg = load_config()
    api = HfApi()
    record = {"info": run_info(), "models": {}}

    for key, spec in [("qwen3", cfg.models.qwen3), ("bge", cfg.models.bge)]:
        sha = api.model_info(spec.name, revision=spec.revision).sha
        local = Path(snapshot_download(spec.name, revision=sha, allow_patterns=ALLOW_PATTERNS))
        files = sorted(p.name for p in local.iterdir())
        size_gb = _dir_size_bytes(local) / 1e9

        record["models"][key] = {
            "name": spec.name,
            "requested_revision": spec.revision,
            "resolved_sha": sha,
            "files": files,
            "size_gb": round(size_gb, 3),
        }
        print(f"{key}: {spec.name} @ {sha}  ({size_gb:.2f} GB, {len(files)} files)")

    out = save_json(record, cfg.paths.results_dir / "model_revisions.json")
    print(f"\nwrote {out}")
    print("\nPaste into configs/default.yaml:")
    for key, m in record["models"].items():
        print(f"  {key}.revision: {m['resolved_sha']}")


if __name__ == "__main__":
    main()