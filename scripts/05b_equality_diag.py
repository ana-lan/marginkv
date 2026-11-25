# scripts/05b_equality_diag.py
"""Phase 3.4 diagnostic: where does the bf16 cached-vs-full gap come from? (variant B)

Subset: the N worst pairs from 05_equality_test.py + M pairs spread over the dataset.
MPS bf16 (config dtype):
  full_self   : full prompt scored alone          vs saved batched full run (04)
  cached_self : cached, built alone + scored alone vs saved batched cached run (05)
  path_single : cached alone vs full alone  (no batching left: pure path/kernel difference)
  batched_gap : saved cached vs saved full   (the gap 05 measured, on this subset)
CPU fp32 on the worst pairs:
  cpu_fp32_path_worst : cached vs full  (correctness exactly where bf16 looked worst)
Output: results/metrics/equality_diag.json
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # must be set before transformers is imported

import argparse

import numpy as np
import torch

from marginkv.config import load_config
from marginkv.kv.cache_ops import build_doc_kvs, score_on_caches, shared_prefix_len, split_doc_first
from marginkv.models.qwen3_reranker import load_model, score_batch
from marginkv.utils.device import get_device, get_dtype
from marginkv.utils.io import load_json, load_pt, run_info, save_json
from marginkv.utils.seed import set_seed


def _dist(x) -> dict:
    x = np.asarray(x)
    return {"max": float(x.max()), "p99": float(np.percentile(x, 99)), "p50": float(np.percentile(x, 50)),
            "mean": float(x.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--worst", type=int, default=32)
    ap.add_argument("--spread", type=int, default=96)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    name, results, artifacts = args.dataset, cfg.paths.results_dir, cfg.paths.artifacts_dir
    art = load_pt(artifacts / "tokenized" / f"{name}_qwen3.pt")
    dfart = load_pt(artifacts / "tokenized" / f"{name}_qwen3_df.pt")
    assert dfart["base_content_sha256"] == art["content_sha256"]
    pb = {**art["pieces"], **dfart["seam_pieces"]}
    top, q_df = art["rerank_candidates"], art["query_df"]
    full_ref = load_json(results / "scores" / f"qwen3_dfB_{name}.json")["scores"]
    cached_ref = load_json(results / "scores" / f"qwen3_dfB_cached_{name}.json")["scores"]

    pairs = [(q, d) for q in art["query_order"] for d in top[q]]
    gap = np.array([abs(cached_ref[q][d] - full_ref[q][d]) for q, d in pairs])
    worst_idx = np.argsort(-gap)[:args.worst]
    worst = [pairs[i] for i in worst_idx]
    ws = set(worst)
    spread = [p for p in pairs[::max(1, len(pairs) // args.spread)] if p not in ws][:args.spread]
    sub, nw = worst + spread, len(worst)
    sl = shared_prefix_len(pb)
    split = {p: split_doc_first(pb, dfart["doc_seam"][p[1]], q_df[p[0]], "B") for p in sub}

    def single_scores(model, device, ps):
        full, cached = [], []
        for p in ps:
            pre, new = split[p]
            full.append(score_batch(model, [pre + new], device)["log_odds"][0])
            kv = build_doc_kvs(model, [pre], device, sl)[0]
            cached.append(score_on_caches(model, [kv], [new], device)["log_odds"][0])
        return np.array(full), np.array(cached)

    def block(x, y):
        d = np.abs(x - y)
        return {"worst": _dist(d[:nw]), "spread": _dist(d[nw:])}

    device = get_device(cfg.runtime.device)
    model = load_model(cfg.models.qwen3.name, cfg.models.qwen3.revision, device, get_dtype(cfg.runtime.dtype))
    f1, c1 = single_scores(model, device, sub)
    fr = np.array([full_ref[q][d] for q, d in sub])
    cr = np.array([cached_ref[q][d] for q, d in sub])
    out = {"full_self": block(f1, fr), "cached_self": block(c1, cr),
           "path_single": block(c1, f1), "batched_gap": block(cr, fr)}
    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    cpu = torch.device("cpu")
    model = load_model(cfg.models.qwen3.name, cfg.models.qwen3.revision, cpu, torch.float32)
    f2, c2 = single_scores(model, cpu, worst)
    out["cpu_fp32_path_worst"] = _dist(np.abs(c2 - f2))

    rows = []
    for i in worst_idx[:10]:
        q, d = pairs[i]
        rows.append({"qid": q, "doc_id": d, "gap": float(gap[i]), "full": full_ref[q][d], "cached": cached_ref[q][d],
                     "prefix_len": len(split[(q, d)][0]), "new_len": len(split[(q, d)][1]),
                     "seam_kind": dfart["seam_kind"][d],
                     "truncated": art["doc_full_len"][d] > cfg.rerank.max_doc_tokens})
    save_json({"run_info": run_info(), "dataset": name, "variant": "B", "worst_n": nw, "spread_n": len(spread),
               "device": str(device), "dtype": cfg.runtime.dtype, **out, "worst_pairs": rows},
              results / "metrics" / "equality_diag.json")

    print(f"\n[{name}] {nw} worst + {len(spread)} spread pairs, single-pair scoring")
    print(f"{'':20s} {'worst max':>9s} {'worst p50':>9s} {'spread max':>10s} {'spread p99':>10s} {'spread mean':>11s}")
    for k in ("full_self", "cached_self", "path_single", "batched_gap"):
        w, s = out[k]["worst"], out[k]["spread"]
        print(f"{k:20s} {w['max']:9.4f} {w['p50']:9.4f} {s['max']:10.4f} {s['p99']:10.4f} {s['mean']:11.5f}")
    c = out["cpu_fp32_path_worst"]
    print(f"\ncpu fp32, cached vs full on the {nw} worst pairs: max {c['max']:.2e}, mean {c['mean']:.2e}")
    print("\nworst pairs:")
    for r in rows:
        print(f"  {r['qid']:>6} {r['doc_id']:>10}  gap {r['gap']:.3f}  full {r['full']:+.3f}  cached {r['cached']:+.3f}  "
              f"prefix {r['prefix_len']}  new {r['new_len']}  {r['seam_kind']}  trunc={r['truncated']}")


if __name__ == "__main__":
    main()