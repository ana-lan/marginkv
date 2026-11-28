# scripts/09b_prune_summary.py
"""Phase 6.5: pruning table with exact packed BF16 sizes, side by side with Phase 5 quantization.

Reads results/metrics/{prune, quant, storage}.json, the 6.4 score files (retention_per_doc = [T, kept]),
and the 6.2 importance artifacts. No model needed.
  - packed bytes per pruned doc = pack.predicted_file_bytes("bf16", kept, shared_len, next_pos=shared_len+T)
    (exact; verified by tests/test_prune.py::test_pack_pruned); corpus total = sum + one shared file
  - keep_all must reproduce the 5.5 BF16 corpus total exactly
  - post-hoc diagnostic (not pre-registered, descriptive only): share of the first 32 doc tokens each arm keeps
Writes results/metrics/prune_summary.json and results/metrics/prune_table.csv.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

from marginkv.config import load_config
from marginkv.kv import pack as K
from marginkv.kv import prune as P
from marginkv.utils.io import load_json, load_pt, run_info, save_csv, save_json

EARLY = 32
MIB = 2 ** 20
GIB = 2 ** 30


def early_coverage(method, r, imp, docs, rpd, seed):
    impd = {"attn_a": imp["imp_a"], "attn_b": imp["imp_b"]}
    vals = []
    for d in docs:
        T = imp["doc_len"][d]
        if method == "attn_h2o":
            keep, _ = P.select_keep("attn_a", T, r, importance=P.importance_h2o(imp["imp_a"][d]))
        elif method.startswith("attn"):
            keep, _ = P.select_keep(method, T, r, importance=impd[method][d])
        elif method == "random":
            keep, _ = P.select_keep(method, T, r, seed=P.doc_seed(seed, d))
        else:
            keep, _ = P.select_keep(method, T, r)
        assert rpd[d] == [T, len(keep)], f"kept count differs from the sweep for {d}"
        n = min(EARLY, T)
        vals.append(float((keep < n).sum()) / n)
    return float(np.mean(vals))


def main():
    cfg = load_config()
    A, R = cfg.paths.artifacts_dir, cfg.paths.results_dir
    prune = load_json(R / "metrics" / "prune.json")
    quant = load_json(R / "metrics" / "quant.json")
    storage = load_json(R / "metrics" / "storage.json")
    rows, out = [], {"run_info": run_info(), "phase": "6.5", "early_tokens": EARLY,
                     "post_hoc": ["early_coverage"], "phase7_selection": prune["summary"]["phase7_selection"],
                     "datasets": {}}

    for ds in cfg.datasets:
        pr, st, qu = prune["datasets"][ds], storage["datasets"][ds], quant["datasets"][ds]
        s, shared_b, ndocs = st["shared_len"], st["shared_file_bytes"], st["num_docs"]
        bf16_total = st["formats"]["bf16"]["corpus_total_bytes"]
        bf16_doc = (bf16_total - shared_b) / ndocs
        imp = load_pt(A / "prune" / f"importance_{ds}.pt")
        dso = {"bf16_doc_bytes_mean": bf16_doc, "arms": {}, "quant": {}}

        keep_rpd = load_json(R / "scores" / f"qwen3_lora_dfB_cached_p6_keep_all_{ds}.json")["retention_per_doc"]
        assert len(keep_rpd) == ndocs, (len(keep_rpd), ndocs)
        keep_total = sum(K.predicted_file_bytes("bf16", k, s, next_pos=s + T) for T, k in keep_rpd.values()) + shared_b
        assert keep_total == bf16_total, f"keep_all packed total {keep_total} != 5.5 BF16 total {bf16_total}"
        print(f"[{ds}] keep_all packed total == 5.5 BF16 corpus total ({bf16_total / GIB:.2f} GiB)")

        for a, e in pr["arms"].items():
            rpd = load_json(R / "scores" / f"qwen3_lora_dfB_cached_p6_{a}_{ds}.json")["retention_per_doc"]
            sizes = [K.predicted_file_bytes("bf16", k, s, next_pos=s + T) for T, k in rpd.values()]
            total = sum(sizes) + shared_b
            cov = early_coverage(e["method"], e["retention"], imp, list(rpd), rpd, cfg.seed)
            v = e["vs_baseline"]
            rec = {"method": e["method"], "retention": e["retention"], "diagnostic": e["diagnostic"],
                   "actual_retention": e["retention_stats"]["actual_retention_tokens"],
                   "forced_frac": e["retention_stats"]["forced_frac"],
                   "d_ndcg10": v["delta_ndcg@10"], "ndcg10_ci": v["ndcg@10_ci95"], "d_mrr10": v["delta_mrr@10"],
                   "verdict": e["verdict"], "mib_per_doc": float(np.mean(sizes)) / MIB,
                   "ratio_vs_bf16": bf16_doc / float(np.mean(sizes)), "corpus_gib": total / GIB,
                   "ratio_vs_bf16_corpus": bf16_total / total, "early_coverage": cov}
            dso["arms"][a] = rec
            rows.append({"dataset": ds, "kind": "prune", "arm": a, **{k: rec[k] for k in (
                "method", "retention", "diagnostic", "actual_retention", "forced_frac", "d_ndcg10")},
                "ci_lo": rec["ndcg10_ci"][0], "ci_hi": rec["ndcg10_ci"][1], "d_mrr10": rec["d_mrr10"],
                "verdict": rec["verdict"], "mib_per_doc": rec["mib_per_doc"], "ratio_vs_bf16": rec["ratio_vs_bf16"],
                "corpus_gib": rec["corpus_gib"], "early_coverage": cov})

        for fmt, f in st["formats"].items():
            q = qu["formats"].get(fmt)
            rec = {"mib_per_doc": f["file_bytes_per_doc_mean"] / MIB, "ratio_vs_bf16": f["ratio_vs_bf16_file"],
                   "corpus_gib": f["corpus_total_gib"],
                   "d_ndcg10": q["vs_bf16"]["delta_ndcg@10"] if q else 0.0,
                   "ndcg10_ci": q["vs_bf16"]["ndcg@10_ci95"] if q else [0.0, 0.0],
                   "d_mrr10": q["vs_bf16"]["delta_mrr@10"] if q else 0.0,
                   "verdict": q["verdict"] if q else "baseline"}
            dso["quant"][fmt] = rec
            rows.append({"dataset": ds, "kind": "quant", "arm": fmt, "method": fmt, "retention": 1.0, "diagnostic": fmt == "int4_tok",
                         "actual_retention": 1.0, "forced_frac": 0.0, "d_ndcg10": rec["d_ndcg10"],
                         "ci_lo": rec["ndcg10_ci"][0], "ci_hi": rec["ndcg10_ci"][1], "d_mrr10": rec["d_mrr10"],
                         "verdict": rec["verdict"], "mib_per_doc": rec["mib_per_doc"], "ratio_vs_bf16": rec["ratio_vs_bf16"],
                         "corpus_gib": rec["corpus_gib"], "early_coverage": None})

        out["datasets"][ds] = dso
        print(f"[{ds}] {'arm':16s} {'MiB/doc':>8s} {'vs BF16':>8s} {'d nDCG@10':>10s} {'95% CI':>20s} {'verdict':>18s} {'early32':>8s}")
        allr = [(r["mib_per_doc"], r) for r in rows if r["dataset"] == ds]
        for _, r in sorted(allr, key=lambda x: -x[0]):
            ec = f"{r['early_coverage']:.3f}" if r["early_coverage"] is not None else "-"
            tag = " (diag)" if r["diagnostic"] else ""
            print(f"[{ds}] {r['arm'] + tag:16s} {r['mib_per_doc']:8.2f} {r['ratio_vs_bf16']:7.2f}x {r['d_ndcg10']:+10.4f} "
                  f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] {r['verdict']:>18s} {ec:>8s}")

    save_json(out, R / "metrics" / "prune_summary.json")
    save_csv(rows, R / "metrics" / "prune_table.csv")
    s7 = out["phase7_selection"]
    print(f"Phase 7 setting: {s7['arm']} (pre-registered rule). wrote prune_summary.json, prune_table.csv")


if __name__ == "__main__":
    main()