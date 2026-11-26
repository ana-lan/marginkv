# src/marginkv/train/data.py
"""MS MARCO training groups for Phase 4 (Tevatron mirror).

A group = 1 positive + (group_size - 1) negatives for one query. Negatives are
picked by stemmed query-term overlap (a hardness proxy, since Tevatron does not
document its mining), sampled from a top pool to limit false negatives.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
from pathlib import Path

import Stemmer

TEVATRON_REPO = "Tevatron/msmarco-passage"
TEVATRON_FILE = "train.jsonl.gz"
TEVATRON_REVISION = "330a160b40a739d4c07d52c7ce562e2c549214c0"

_STOP = frozenset(
    "a an and are as at be by can did do does for from has have how in is it its "
    "of on or that the this to was were what when where which who why will with".split()
)
_WORD = re.compile(r"\w+")
_STEMMER = Stemmer.Stemmer("english")


def normalize_text(s: str) -> str:
    return " ".join(s.lower().split())


def terms(s: str) -> set[str]:
    toks = [t for t in _WORD.findall(s.lower()) if t not in _STOP]
    return set(_STEMMER.stemWords(toks))


def overlap(q_terms: set[str], text: str) -> float:
    """Fraction of query terms present in text."""
    if not q_terms:
        return 0.0
    return len(q_terms & terms(text)) / len(q_terms)


def passage_text(p: dict) -> str:
    title = (p.get("title") or "").strip()
    text = (p.get("text") or "").strip()
    if title and title != "-":
        return f"{title} {text}"
    return text


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def content_sha256(obj) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def count_lines(path: Path) -> int:
    n = 0
    with gzip.open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def read_selected(path: Path, wanted: set[int]) -> dict[int, dict]:
    """Parse only the line indices in `wanted`."""
    out: dict[int, dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in wanted:
                out[i] = json.loads(line)
                if len(out) == len(wanted):
                    break
    return out


def select_group(row: dict, group_size: int, pool: int, seed: int):
    """Return (group_pids, texts{pid: text}, hardness) or (None, reason, None).

    group_pids[0] is the positive. Negatives: dedup by id and text, never equal
    to any positive's id or text, top `pool` by overlap, then a seeded sample
    of group_size - 1 kept in overlap order.
    """
    k = group_size - 1
    positives = [p for p in row.get("positive_passages", []) if passage_text(p)]
    if not positives:
        return None, "no_positive", None
    pos = positives[0]
    pos_ids = {str(p["docid"]) for p in positives}
    pos_norm = {normalize_text(passage_text(p)) for p in positives}

    negs, seen_ids, seen_norm = [], set(), set()
    for p in row.get("negative_passages", []):
        pid, t = str(p["docid"]), passage_text(p)
        n = normalize_text(t)
        if not t or pid in pos_ids or n in pos_norm or pid in seen_ids or n in seen_norm:
            continue
        seen_ids.add(pid)
        seen_norm.add(n)
        negs.append((pid, t))
    if len(negs) < k:
        return None, "few_negatives", None

    q_terms = terms(row["query"])
    scored = sorted(((overlap(q_terms, t), pid, t) for pid, t in negs), key=lambda x: (-x[0], x[1]))
    top = scored[: max(pool, k)]
    rng = random.Random(f"{seed}-{row['query_id']}")
    chosen = sorted(rng.sample(top, k), key=lambda x: (-x[0], x[1]))

    pos_pid, pos_t = str(pos["docid"]), passage_text(pos)
    texts = {pos_pid: pos_t, **{pid: t for _, pid, t in chosen}}
    hard = {
        "pos_overlap": overlap(q_terms, pos_t),
        "neg_overlap_chosen": sum(s for s, _, _ in chosen) / k,
        "neg_overlap_all": sum(s for s, _, _ in scored) / len(scored),
        "max_neg_ge_pos": float(chosen[0][0] >= overlap(q_terms, pos_t)),
    }
    return [pos_pid] + [pid for _, pid, _ in chosen], texts, hard