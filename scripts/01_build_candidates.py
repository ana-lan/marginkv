# scripts/01_build_candidates.py
"""Phase 1: load BEIR, BM25 top-k, save candidates once, report Recall@100 and dataset stats.

Outputs (all embed run_info()):
  results/candidates/{ds}.json   BM25 top-k per query, rank order, query file order
  results/metrics/bm25.json      nDCG@10, MRR@10, Recall@100 (mean + per query)
  results/datasets.json          counts, relevance levels, sources, file hashes, BM25 diagnostics
"""
from __future__ import annotations

import argparse
from dataclasses import asdict

from marginkv.config import load_config
from marginkv.data.beir_loader import dataset_stats, load_beir
from marginkv.eval.metrics import evaluate_ranked
from marginkv.retrieval.bm25 import BM25Params, retrieve_all
from marginkv.utils.io import load_json, run_info, save_json
from marginkv.utils.seed import set_seed


def _round_run(run: dict) -> dict:
    # Fixed rounding so reruns produce byte-identical files.
    return {qid: [[d, round(s, 6)] for d, s in hits] for qid, hits in run.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="overwrite saved candidates even if they differ")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    if cfg.retrieval.method != "bm25":
        raise ValueError(f"retrieval.method must be bm25, got {cfg.retrieval.method}")

    k = cfg.retrieval.pool_size
    results = cfg.paths.results_dir
    params = BM25Params()
    info = run_info()

    stats_out, metrics_out, summary = {}, {}, []
    for name in cfg.datasets:
        ds = load_beir(name, split=cfg.split, data_dir=cfg.paths.data_dir)
        run, bm25_info = retrieve_all(ds, k=k, params=params)
        cands = _round_run(run)

        path = results / "candidates" / f"{name}.json"
        if path.exists() and not args.force:
            if load_json(path)["candidates"] != cands:
                raise SystemExit(f"[{name}] new candidates differ from {path}. Investigate, or rerun with --force.")
            print(f"[{name}] candidates unchanged: {path}")
        else:
            save_json({
                "run_info": info,
                "dataset": name,
                "split": cfg.split,
                "source": {key: ds.manifest[key] for key in ("source_repo", "source_revision", "files_sha256")},
                "bm25": bm25_info,
                "query_order": list(ds.queries),
                "candidates": cands,  # {qid: [[doc_id, score], ...]}, index 0 = rank 1
            }, path)
            print(f"[{name}] saved {path}")

        # Evaluate what is actually on disk.
        saved = load_json(path)["candidates"]
        m = evaluate_ranked(saved, ds.qrels)
        metrics_out[name] = m

        st = dataset_stats(ds)
        st["files_sha256"] = ds.manifest["files_sha256"]
        st["bm25_queries_no_known_terms"] = len(bm25_info["queries_with_no_known_terms"])
        st["bm25_queries_no_known_terms_ids"] = bm25_info["queries_with_no_known_terms"]
        st["bm25_queries_fewer_than_k_matches"] = bm25_info["queries_with_fewer_than_k_matches"]
        stats_out[name] = st
        summary.append((name, st["num_docs"], st["num_queries"], m["mean"]))

    save_json({
        "run_info": info,
        "method": "bm25",
        "bm25": asdict(params),
        "pool_size": k,
        "split": cfg.split,
        "datasets": metrics_out,
    }, results / "metrics" / "bm25.json")
    save_json({"run_info": info, "datasets": stats_out}, results / "datasets.json")

    print(f"\n{'dataset':10s} {'docs':>6s} {'queries':>8s} {'nDCG@10':>8s} {'MRR@10':>8s} {'R@100':>8s}")
    for name, nd, nq, mean in summary:
        print(f"{name:10s} {nd:6d} {nq:8d} {mean['ndcg@10']:8.4f} {mean['mrr@10']:8.4f} {mean['recall@100']:8.4f}")


if __name__ == "__main__":
    main()