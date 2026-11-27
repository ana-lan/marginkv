# src/marginkv/data/beir_loader.py
"""Download BEIR datasets (pinned HF mirror, sha256 recorded) and load corpus, queries, qrels.

Source: the mteb/* Hugging Face mirrors keep the official BEIR layout
(corpus.jsonl, queries.jsonl, qrels/{split}.tsv). Used because the original
UKP zip server was unreachable on 2026-11-24. Counts are checked against BEIR.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

HF_MIRRORS = {
    "scifact": ("mteb/scifact", "cf10ab6856b15b0e670ef8ae5dae4e266c12d035"),
    "nfcorpus": ("mteb/nfcorpus", "52ac3f19d3449632d9f00aab0ad34a110fc03816"),
}
SUPPORTED = tuple(HF_MIRRORS)
REQUIRED_FILES = ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")


@dataclass(frozen=True)
class BeirDataset:
    name: str
    split: str
    corpus: dict[str, dict[str, str]]  # doc_id -> {"title": ..., "text": ...}
    queries: dict[str, str]  # query_id -> text, only queries in this split, file order
    qrels: dict[str, dict[str, int]]  # query_id -> {doc_id: relevance}
    manifest: dict


# ---------- file helpers ----------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json_atomic(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


# ---------- download ----------

def download_beir(name: str, data_dir: str | Path = "data", force: bool = False) -> dict:
    """Fetch one dataset at a pinned revision into data/{name}/. Returns its manifest."""
    if name not in HF_MIRRORS:
        raise ValueError(f"Unsupported dataset {name!r}, expected one of {SUPPORTED}")
    repo_id, revision = HF_MIRRORS[name]
    ds_dir = Path(data_dir) / name
    manifest_path = ds_dir / "manifest.json"

    # Already downloaded: verify every file against its recorded hash.
    if not force and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_revision") != revision:
            raise RuntimeError(f"[{name}] manifest revision differs from pinned one. Rerun with force=True.")
        for rel, sha in manifest["files_sha256"].items():
            p = ds_dir / rel
            if not p.exists() or _sha256(p) != sha:
                raise RuntimeError(f"[{name}] {rel} missing or changed since download. Rerun with force=True.")
        return manifest

    from huggingface_hub import hf_hub_download  # lazy: only needed on first download

    files = {}
    for rel in REQUIRED_FILES:
        print(f"[{name}] fetching {repo_id}@{revision[:8]}:{rel}")
        cached = hf_hub_download(repo_id=repo_id, filename=rel, repo_type="dataset", revision=revision)
        dst = ds_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copyfile(cached, tmp)  # real copy in data/, not a symlink into the HF cache
        os.replace(tmp, dst)
        files[rel] = _sha256(dst)

    manifest = {
        "dataset": name,
        "source": "huggingface",
        "source_repo": repo_id,
        "source_revision": revision,
        "files_sha256": files,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_json_atomic(manifest, manifest_path)
    return manifest


# ---------- readers ----------

def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_corpus(path: Path) -> dict[str, dict[str, str]]:
    corpus: dict[str, dict[str, str]] = {}
    for row in _read_jsonl(path):
        doc_id = str(row["_id"])
        if doc_id in corpus:
            raise ValueError(f"Duplicate doc id {doc_id} in {path}")
        corpus[doc_id] = {"title": row.get("title") or "", "text": row.get("text") or ""}
    return corpus


def _read_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}  # dicts keep insertion order = file order
    for row in _read_jsonl(path):
        qid = str(row["_id"])
        if qid in queries:
            raise ValueError(f"Duplicate query id {qid} in {path}")
        queries[qid] = row["text"]
    return queries


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        if [h.strip() for h in header[:3]] != ["query-id", "corpus-id", "score"]:
            raise ValueError(f"Unexpected qrels header in {path}: {header}")
        for row in reader:
            if not row:
                continue
            qid, did, score = row[0].strip(), row[1].strip(), int(row[2])
            prev = qrels.setdefault(qid, {}).get(did)
            if prev is not None and prev != score:
                raise ValueError(f"Conflicting qrels for ({qid}, {did}): {prev} vs {score}")
            qrels[qid][did] = score
    return qrels


# ---------- public API ----------

def load_beir(name: str, split: str = "test", data_dir: str | Path = "data",
              download: bool = True) -> BeirDataset:
    ds_dir = Path(data_dir) / name
    if download:
        manifest = download_beir(name, data_dir)
    else:
        manifest = json.loads((ds_dir / "manifest.json").read_text(encoding="utf-8"))

    corpus = _read_corpus(ds_dir / "corpus.jsonl")
    all_queries = _read_queries(ds_dir / "queries.jsonl")
    qrels = _read_qrels(ds_dir / "qrels" / f"{split}.tsv")

    missing = [q for q in qrels if q not in all_queries]
    if missing:
        raise ValueError(f"[{name}] {len(missing)} qrel query ids have no query text, e.g. {missing[:3]}")

    # Keep only queries judged in this split, in original file order.
    queries = {qid: text for qid, text in all_queries.items() if qid in qrels}
    return BeirDataset(name, split, corpus, queries, qrels, manifest)


def dataset_stats(ds: BeirDataset) -> dict:
    levels = Counter(rel for judged in ds.qrels.values() for rel in judged.values())
    pos_per_q = [sum(1 for r in judged.values() if r > 0) for judged in ds.qrels.values()]
    judged_docs = {d for judged in ds.qrels.values() for d in judged}
    doc_words = [len((d["title"] + " " + d["text"]).split()) for d in ds.corpus.values()]
    q_words = [len(t.split()) for t in ds.queries.values()]
    return {
        "dataset": ds.name,
        "split": ds.split,
        "num_docs": len(ds.corpus),
        "num_queries": len(ds.queries),
        "num_qrels": sum(len(j) for j in ds.qrels.values()),
        "relevance_levels": {str(k): v for k, v in sorted(levels.items())},
        "positives_per_query_mean": round(sum(pos_per_q) / len(pos_per_q), 3),
        "positives_per_query_min": min(pos_per_q),
        "positives_per_query_max": max(pos_per_q),
        "queries_without_positive": sum(1 for p in pos_per_q if p == 0),
        "qrel_docs_missing_from_corpus": len(judged_docs - ds.corpus.keys()),
        "docs_empty_title": sum(1 for d in ds.corpus.values() if not d["title"].strip()),
        "docs_empty_text": sum(1 for d in ds.corpus.values() if not d["text"].strip()),
        "doc_words_mean": round(sum(doc_words) / len(doc_words), 1),
        "query_words_mean": round(sum(q_words) / len(q_words), 1),
        "source_repo": ds.manifest["source_repo"],
        "source_revision": ds.manifest["source_revision"],
    }


if __name__ == "__main__":
    names = sys.argv[1:] or list(SUPPORTED)
    for n in names:
        print(json.dumps(dataset_stats(load_beir(n)), indent=2))