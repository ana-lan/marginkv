# src/marginkv/eval/paired.py
"""Paired per-query comparisons and verdicts (Phase 5 on). Same math as the helpers inside
scripts/07a_noise_floor.py, which keeps its own copies so the 5.2 output stays reproducible as-is."""
from __future__ import annotations

import numpy as np

from marginkv.eval.rerank import rerank_merge

N_BOOT = 10000
ONE_POINT = 0.01


def paired_ci(a, b, seed, n_boot: int = N_BOOT):
    """Mean of b - a and its percentile bootstrap 95% CI (paired query resamples)."""
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), [float(lo), float(hi)]


def wlt(a, b, eps: float = 1e-9):
    w = sum(1 for x, y in zip(a, b) if y > x + eps)
    l = sum(1 for x, y in zip(a, b) if y < x - eps)
    return [w, l, len(a) - w - l]


def compare(pq_a: dict, pq_b: dict, qids, seed) -> dict:
    """b vs a on per-query nDCG@10 and MRR@10."""
    out = {}
    for m in ("ndcg@10", "mrr@10"):
        a = [pq_a[q][m] for q in qids]
        b = [pq_b[q][m] for q in qids]
        out[f"delta_{m}"], out[f"{m}_ci95"] = paired_ci(a, b, seed)
    out["wlt_ndcg@10"] = wlt([pq_a[q]["ndcg@10"] for q in qids], [pq_b[q]["ndcg@10"] for q in qids])
    return out


def score_agreement(cands, sa, sb, qids, depth) -> dict:
    diffs, flips, tot = [], 0, 0
    for q in qids:
        docs = list(sa[q])
        a = [sa[q][d] for d in docs]
        b = [sb[q][d] for d in docs]
        diffs += [abs(x - y) for x, y in zip(a, b)]
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                tot += 1
                flips += int(np.sign(a[i] - a[j]) != np.sign(b[i] - b[j]))
    ra, rb = rerank_merge(cands, sa, depth), rerank_merge(cands, sb, depth)
    same = sum(1 for q in qids if ra[q][:depth] == rb[q][:depth])
    return {"max_abs": float(max(diffs)), "p99_abs": float(np.percentile(diffs, 99)),
            "mean_abs": float(np.mean(diffs)), "pair_flip_rate": flips / tot,
            "identical_topk_order": [same, len(qids)]}


def verdict(delta: float, ci, margin: float, one_point: float = ONE_POINT) -> str:
    """Pre-registered in 5.2: CI inside +-margin -> no_measurable_loss; point >= -0.01 -> within_1_point."""
    if -margin <= ci[0] and ci[1] <= margin:
        return "no_measurable_loss"
    if delta >= -one_point:
        return "within_1_point"
    return "loss"