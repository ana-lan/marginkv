# src/marginkv/eval/rerank.py
"""Rerank the BM25 top-`depth` with model scores, append the BM25 tail, evaluate.

Final list per query = top-`depth` docs sorted by (score desc, BM25 rank asc), then BM25 ranks
depth+1 .. pool in their original order. The tie rule is explicit so the order never depends
on dict or iteration order.
"""
from __future__ import annotations

import math

from marginkv.eval.metrics import evaluate_ranked


def _head_tail(hits: list, depth: int, qid: str) -> tuple[list[str], list[str]]:
    if len(hits) < depth:
        raise ValueError(f"query {qid}: {len(hits)} candidates < depth {depth}")
    ids = [h[0] if isinstance(h, (list, tuple)) else h for h in hits]
    if len(set(ids)) != len(ids):
        raise ValueError(f"query {qid}: duplicate doc ids in candidates")
    return ids[:depth], ids[depth:]


def rerank_merge(candidates: dict, scores: dict, depth: int) -> dict[str, list[str]]:
    """candidates: {qid: [[doc_id, bm25_score], ...]} (index 0 = rank 1) or {qid: [doc_id, ...]}.
    scores: {qid: {doc_id: float}} covering exactly the top-`depth` docs of every query.
    Returns {qid: [doc_id, ...]} with the same length as the candidate list."""
    missing_q = set(candidates) - set(scores)
    extra_q = set(scores) - set(candidates)
    if missing_q or extra_q:
        raise ValueError(f"score queries differ from candidates: missing {sorted(missing_q)[:5]}, "
                         f"extra {sorted(extra_q)[:5]}")
    ranked = {}
    for qid, hits in candidates.items():
        head, tail = _head_tail(hits, depth, qid)
        s = scores[qid]
        if set(s) != set(head):
            raise ValueError(f"query {qid}: scored docs != top-{depth} "
                             f"(missing {sorted(set(head) - set(s))[:3]}, extra {sorted(set(s) - set(head))[:3]})")
        for d, v in s.items():
            if not math.isfinite(float(v)):
                raise ValueError(f"query {qid}: non-finite score {v} for doc {d}")
        bm25_rank = {d: r for r, d in enumerate(head)}
        new_head = sorted(head, key=lambda d: (-float(s[d]), bm25_rank[d]))
        ranked[qid] = new_head + tail
    return ranked


def identity_scores(candidates: dict, depth: int) -> dict:
    """Scores that reproduce BM25 order: -rank. Used as a sanity check."""
    out = {}
    for qid, hits in candidates.items():
        head, _ = _head_tail(hits, depth, qid)
        out[qid] = {d: -float(r) for r, d in enumerate(head)}
    return out


def rerank_diagnostics(candidates: dict, scores: dict, depth: int) -> dict:
    """How much the reranker changed the head, and how often its scores tie."""
    n = len(candidates)
    tie_queries = tied_docs = top1_changed = 0
    moved = 0.0
    for qid, hits in candidates.items():
        head, _ = _head_tail(hits, depth, qid)
        vals = [float(scores[qid][d]) for d in head]
        n_tied = len(vals) - len(set(vals))
        tied_docs += n_tied
        tie_queries += n_tied > 0
        bm25_rank = {d: r for r, d in enumerate(head)}
        new_head = sorted(head, key=lambda d: (-float(scores[qid][d]), bm25_rank[d]))
        top1_changed += new_head[0] != head[0]
        moved += sum(abs(i - bm25_rank[d]) for i, d in enumerate(new_head)) / depth
    return {
        "num_queries": n,
        "queries_with_score_ties": tie_queries,
        "duplicate_score_values": tied_docs,
        "top1_changed_frac": round(top1_changed / n, 4),
        "mean_abs_rank_shift": round(moved / n, 4),
    }


def evaluate_rerank(candidates: dict, scores: dict, qrels: dict, depth: int) -> dict:
    """Merge, evaluate, and attach diagnostics. Returns evaluate_ranked's dict plus 'diagnostics'."""
    ranked = rerank_merge(candidates, scores, depth)
    m = evaluate_ranked(ranked, qrels)
    m["diagnostics"] = rerank_diagnostics(candidates, scores, depth)
    return m