# src/marginkv/eval/metrics.py
"""nDCG@10, MRR@10, Recall@100 via pytrec_eval, evaluating OUR rank order exactly."""
from __future__ import annotations

from collections.abc import Sequence

import pytrec_eval

METRICS = ("ndcg@10", "mrr@10", "recall@100")


def _ids(hits: Sequence) -> list[str]:
    """Accept [doc_id, ...] or [(doc_id, score), ...]."""
    return [h[0] if isinstance(h, (tuple, list)) else h for h in hits]


def ranked_to_run(ranked: dict[str, Sequence]) -> dict[str, dict[str, float]]:
    """Ranked lists -> strictly decreasing scores, so pytrec_eval's re-sort keeps our order."""
    run = {}
    for qid, hits in ranked.items():
        ids = _ids(hits)
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate doc ids in ranking for query {qid}")
        n = len(ids)
        run[qid] = {d: float(n - i) for i, d in enumerate(ids)}
    return run


def evaluate_ranked(ranked: dict[str, Sequence], qrels: dict[str, dict[str, int]]) -> dict:
    """Returns {"mean": {metric: x}, "per_query": {qid: {metric: x}}, "num_queries": n}."""
    missing = qrels.keys() - ranked.keys()
    extra = ranked.keys() - qrels.keys()
    if missing or extra:
        raise ValueError(f"Run/qrels query mismatch: {len(missing)} missing, {len(extra)} extra")

    full = ranked_to_run(ranked)
    top10 = ranked_to_run({q: _ids(h)[:10] for q, h in ranked.items()})

    main = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.100"}).evaluate(full)
    rr = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"}).evaluate(top10)

    per_query = {}
    for qid in qrels:
        if qid not in main or qid not in rr:
            raise RuntimeError(f"pytrec_eval dropped query {qid}")
        per_query[qid] = {
            "ndcg@10": main[qid]["ndcg_cut_10"],
            "mrr@10": rr[qid]["recip_rank"],
            "recall@100": main[qid]["recall_100"],
        }
    n = len(per_query)
    mean = {m: sum(v[m] for v in per_query.values()) / n for m in METRICS}
    return {"mean": mean, "per_query": per_query, "num_queries": n}