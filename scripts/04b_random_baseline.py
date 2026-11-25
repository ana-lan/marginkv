# scripts/04b_random_baseline.py
"""Phase 3.5: random-order baseline for the reranked top 20 (chance level for Table 1).

Each rep gives the top-`depth` docs of every query random scores (BM25 tail 21-100 unchanged),
then evaluates exactly like the rerankers. R reps, seeded.
Output: results/metrics/random_baseline.json
"""
from __future__ import annotations

import argparse

import numpy as np

from marginkv.config import load_config
from marginkv.data.beir_loader import load_beir
from marginkv.eval.rerank import evaluate_rerank
from marginkv.utils.io import load_json, run_info, save_json


def _summ(x) -> dict:
    x = np.asarray(x)
    return {"mean": float(x.mean()), "std": float(x.std()),
            "p2.5": float(np.percentile(x, 2.5)), "p97.5": float(np.percentile(x, 97.5))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config(args.config)
    depth, results = cfg.rerank.depth, cfg.paths.results_dir
    rng = np.random.default_rng(cfg.seed)

    out = {}
    for name in cfg.datasets:
        ds = load_beir(name, split=cfg.split, data_dir=cfg.paths.data_dir)
        cand = load_json(results / "candidates" / f"{name}.json")
        qids, cs = cand["query_order"], cand["candidates"]
        qrels = {q: ds.qrels[q] for q in qids}
        nd, mr = [], []
        for _ in range(args.reps):
            scores = {q: {d: float(s) for (d, _), s in zip(cs[q][:depth], rng.random(depth))} for q in qids}
            m = evaluate_rerank(cs, scores, qrels, depth)["mean"]
            nd.append(m["ndcg@10"])
            mr.append(m["mrr@10"])
        out[name] = {"reps": args.reps, "ndcg@10": _summ(nd), "mrr@10": _summ(mr)}

    save_json({"run_info": run_info(), "depth": depth, "seed": cfg.seed, "datasets": out},
              results / "metrics" / "random_baseline.json")
    for name, m in out.items():
        n, r = m["ndcg@10"], m["mrr@10"]
        print(f"{name:9s} random top-{depth}: nDCG@10 {n['mean']:.4f} (95% {n['p2.5']:.4f}-{n['p97.5']:.4f})  "
              f"MRR@10 {r['mean']:.4f} (95% {r['p2.5']:.4f}-{r['p97.5']:.4f})")


if __name__ == "__main__":
    main()