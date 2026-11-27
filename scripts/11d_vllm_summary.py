# scripts/11d_vllm_summary.py
"""Phase 8a.6: summarize the vLLM runs (results/vllm/runs/*.json) into results/metrics/vllm.json
and results/vllm/table3.csv. Reruns the 8a.1 LRU prefix-cache simulation at the measured block size
and capacity per setting, and fits per-request latency vs uncached tokens at concurrency 1.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
from collections import OrderedDict

import numpy as np

from marginkv.config import load_config
from marginkv.utils.io import load_json, load_pt, run_info, save_csv, save_json

SETTINGS = ("qf", "df", "df_fp8")
ORDER = {"qf": "qf", "df": "df", "df_fp8": "df"}
TRACES = ("t3", "t1", "t2")
CONC = (16, 4, 1)
MODES = ("cold", "warm")


def simulate(prompts, spans, cap, block, cache=None):
    """Sequential block-level prefix cache, LRU over cached full blocks (same rules as 8a.1)."""
    cache = OrderedDict() if cache is None else cache
    hit_tok = tot = doc_hit = doc_tot = 0
    for ids, (a, b) in zip(prompts, spans):
        n = len(ids)
        hs, h = [], None
        for k in range(n // block):
            h = hash((h, tuple(ids[k * block:(k + 1) * block])))
            hs.append(h)
        max_hit = (n - 1) // block
        hit = 0
        while hit < max_hit and hs[hit] in cache:
            hit += 1
        ht = hit * block
        hit_tok += ht
        tot += n
        doc_hit += max(0, min(ht, b) - a)
        doc_tot += b - a
        for x in reversed(hs):
            cache[x] = True
            cache.move_to_end(x)
        while cap is not None and len(cache) > cap:
            cache.popitem(last=False)
    return {"hit_frac": hit_tok / tot, "doc_tokens_reused_frac": doc_hit / doc_tot}, cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    args = ap.parse_args()
    cfg = load_config()
    res, arts = cfg.paths.results_dir, cfg.paths.artifacts_dir / "traces"
    datasets = args.datasets or list(cfg.datasets)
    runs_dir = res / "vllm" / "runs"

    runs, rows, quality = {}, [], []
    for s in SETTINGS:
        for ds in datasets:
            for t in TRACES:
                for c in CONC:
                    for m in MODES:
                        p = runs_dir / f"{s}_{t}_{ds}_c{c}_{m}.json"
                        assert p.exists(), f"missing run {p.name}"
                        d = load_json(p)
                        runs[(s, ds, t, c, m)] = d
                        sm, cc = d["summary"], d["server"]["cache_config"]
                        assert sm["missing_yes_no"] == 0 and sm["errors"] == 0, p.name
                        rows.append({"setting": s, "dataset": ds, "trace": t, "concurrency": c, "mode": m,
                                     "n": sm["n"], "req_per_s": round(sm["req_per_s"], 1),
                                     "prompt_tok_per_s": round(sm["prompt_tok_per_s"]),
                                     "p50_ms": round(sm["latency_ms"]["p50"], 2), "p95_ms": round(sm["latency_ms"]["p95"], 2),
                                     "hit_frac": round(sm["hit_frac"], 4),
                                     "doc_tokens_reused_frac": round(sm["doc_tokens_reused_frac"], 4),
                                     "metric_hit_frac": round(sm["metrics_delta"]["hit_frac"], 4),
                                     "block_size": int(cc["block_size"]), "num_gpu_blocks": int(cc["num_gpu_blocks"]),
                                     "kv_cache_tokens": int(cc["kv_cache_size_tokens"]), "cache_dtype": cc["cache_dtype"],
                                     "vllm": d["server"]["version"]})
                        if "quality" in d:
                            q = d["quality"]
                            row = {"setting": s, "dataset": ds, "concurrency": c,
                                   "ndcg@10": q["mean"]["ndcg@10"], "mrr@10": q["mean"]["mrr@10"],
                                   "baseline": q["baseline"], "baseline_ndcg@10": q["baseline_mean"]["ndcg@10"],
                                   "delta_ndcg@10": q["vs_baseline"]["delta_ndcg@10"],
                                   "ndcg@10_ci95": q["vs_baseline"]["ndcg@10_ci95"],
                                   "delta_mrr@10": q["vs_baseline"]["delta_mrr@10"],
                                   "wlt_ndcg@10": q["vs_baseline"]["wlt_ndcg@10"], "G4_pass": q["G4_pass"],
                                   "score_ties_queries": q["diagnostics"].get("queries_with_score_ties")}
                            if "vs_vllm_df_bf16" in q:
                                v = q["vs_vllm_df_bf16"]
                                row["fp8_vs_vllm_bf16"] = {k: v[k] for k in ("delta_ndcg@10", "ndcg@10_ci95", "delta_mrr@10",
                                                                             "wlt_ndcg@10", "margin", "verdict")}
                                row["fp8_vs_vllm_bf16"]["scores"] = v["scores"]
                            quality.append(row)

    # simulation at the measured block size / capacity vs measured c=1 (sequential) and c=16
    sim = []
    traces = {(ds, t): load_pt(arts / f"{ds}_{t}.pt") for ds in datasets for t in TRACES}
    for s in SETTINGS:
        for ds in datasets:
            for t in TRACES:
                tr = traces[(ds, t)]
                o = ORDER[s]
                d1 = runs[(s, ds, t, 1, "cold")]["server"]["cache_config"]
                block, cap = int(d1["block_size"]), int(d1["num_gpu_blocks"])
                cold, cache = simulate(tr[f"{o}_ids"], tr[f"{o}_span"], cap, block)
                warm, _ = simulate(tr[f"{o}_ids"], tr[f"{o}_span"], cap, block, cache)
                inf, _ = simulate(tr[f"{o}_ids"], tr[f"{o}_span"], None, block)
                entry = {"setting": s, "dataset": ds, "trace": t, "block_size": block, "cap_blocks": cap,
                         "sim_cold": cold, "sim_warm": warm, "sim_unbounded": inf, "measured": {}}
                for c in CONC:
                    for m in MODES:
                        sm = runs[(s, ds, t, c, m)]["summary"]
                        entry["measured"][f"c{c}_{m}"] = {"hit_frac": sm["hit_frac"],
                                                          "doc_tokens_reused_frac": sm["doc_tokens_reused_frac"]}
                m1 = entry["measured"]
                entry["abs_err_c1"] = {"cold": abs(m1["c1_cold"]["hit_frac"] - cold["hit_frac"]),
                                       "warm": abs(m1["c1_warm"]["hit_frac"] - warm["hit_frac"])}
                sim.append(entry)

    # per-request latency vs uncached tokens at c=1
    latency_fit = {}
    for s in SETTINGS:
        xs, ys = [], []
        for ds in datasets:
            for t in TRACES:
                lens = [len(p) for p in traces[(ds, t)][f"{ORDER[s]}_ids"]]
                for m in MODES:
                    pr = runs[(s, ds, t, 1, m)]["per_request"]
                    assert len(pr["latency_ms"]) == len(lens)
                    xs += [(n - c) / 1000.0 for n, c in zip(lens, pr["cached"])]
                    ys += pr["latency_ms"]
        xs, ys = np.asarray(xs), np.asarray(ys)
        slope, intercept = np.polyfit(xs, ys, 1)
        latency_fit[s] = {"n_requests": int(xs.size), "intercept_ms": float(intercept),
                          "ms_per_1k_uncached_tokens": float(slope), "pearson_r": float(np.corrcoef(xs, ys)[0, 1]),
                          "uncached_tokens_k": {"p5": float(np.percentile(xs, 5)), "p95": float(np.percentile(xs, 95))},
                          "latency_ms": {"p50": float(np.percentile(ys, 50)), "p95": float(np.percentile(ys, 95))}}

    # gates, with G3 re-evaluated at the actual block size
    gates = {}
    for s in SETTINGS:
        g = load_json(res / "vllm" / f"gate_{s}.json")
        entry = {"G1_pass": g["G1"]["pass"], "G2": None, "server": {k: g["server"]["cache_config"].get(k) for k in
                 ("block_size", "num_gpu_blocks", "kv_cache_size_tokens", "cache_dtype")},
                 "logprobs": g["logprobs"], "logprobs_mode": g["logprobs_mode"]}
        if g["G2"]:
            entry["G2"] = {"vllm_mean_err": g["G2"]["vllm_vs_fp32"]["mean"], "hf_bf16_mean_err": g["G2"]["hf_bf16_vs_fp32"]["mean"],
                           "ratio": g["G2"]["ratio"], "flag": g["G2"]["flag_ratio_gt_2"],
                           "flip_rate": g["G2"]["pair_flip_rate_vs_fp32"], "tie_groups": g["G2"]["tie_groups"]}
        if g["G3"]:
            B = int(g["server"]["cache_config"]["block_size"])
            pr = g["G3"]["probes"]
            entry["G3"] = {"literal_rule_16_pass": g["G3"]["pass"], "actual_block_size": B,
                           "pass_at_actual_block": all(p["a_cached"] == 0 and p["b_cached"] >= (p["cache_len"] // B) * B for p in pr),
                           "exact_at_actual_block": all(p["b_cached"] == (p["cache_len"] // B) * B for p in pr),
                           "n_probes": len(pr), "usage_cached_reported": g["G3"]["usage_cached_reported"]}
        gates[s] = entry

    # trace 3 headline (cold)
    headline = {}
    for ds in datasets:
        headline[ds] = {}
        for s in SETTINGS:
            r16 = runs[(s, ds, "t3", 16, "cold")]["summary"]
            r1 = runs[(s, ds, "t3", 1, "cold")]["summary"]
            q = next(x for x in quality if x["setting"] == s and x["dataset"] == ds and x["concurrency"] == 16)
            headline[ds][s] = {"hit_frac": r16["hit_frac"], "doc_tokens_reused_frac": r16["doc_tokens_reused_frac"],
                               "c16_req_per_s": r16["req_per_s"], "c16_p50_ms": r16["latency_ms"]["p50"],
                               "c16_p95_ms": r16["latency_ms"]["p95"], "c1_p50_ms": r1["latency_ms"]["p50"],
                               "c1_p95_ms": r1["latency_ms"]["p95"], "ndcg@10": q["ndcg@10"],
                               "kv_cache_tokens": runs[(s, ds, "t3", 16, "cold")]["server"]["cache_config"]["kv_cache_size_tokens"]}

    stats = load_json(res / "metrics" / "trace_stats.json")
    repeat = {ds: {k: stats["datasets"][ds]["traces"]["t3"]["repeats"][k]
                   for k in ("repeat_frac_id", "repeat_frac_content", "reuse_distance_requests")} for ds in datasets}

    out = {"run_info": run_info(), "phase": "8a", "vllm": rows[0]["vllm"],
           "capacity": {s: {k: next(r[k] for r in rows if r["setting"] == s) for k in ("block_size", "num_gpu_blocks", "kv_cache_tokens")} for s in SETTINGS},
           "trace3_repeats": repeat, "trace3_headline": headline, "gates": gates,
           "quality": quality, "simulation_vs_measured": sim, "latency_fit_c1": latency_fit,
           "table3": rows,
           "notes": ["block size 64 chosen by vLLM 0.34 (not user-specified); G3 pre-registered at 16",
                     "c=16 throughput varies run to run by up to ~30% at equal hit rates; compare hit rates and c=1 latency",
                     "fp8 KV uses vLLM defaults (no calibrated scales; scale 1.0); FlashInfer backend, sm89",
                     "logprobs: processed_logprobs + allowed_token_ids [no, yes], --generation-config vllm",
                     "max_num_batched_tokens = vLLM default 2048"]}
    save_json(out, res / "metrics" / "vllm.json")
    save_csv(rows, res / "vllm" / "table3.csv")

    # printout
    print("trace 3, cold (hit = prompt tokens from cache; doc = doc tokens reused):")
    for ds in datasets:
        print(f"  {ds}: repeat id {repeat[ds]['repeat_frac_id']:.3f}, reuse distance p50 {repeat[ds]['reuse_distance_requests']['p50']}")
        for s in SETTINGS:
            h = headline[ds][s]
            print(f"    {s:7s} kv {h['kv_cache_tokens']:>7s} tok | hit {h['hit_frac']:.3f} doc {h['doc_tokens_reused_frac']:.3f} | "
                  f"c16 {h['c16_req_per_s']:.0f} req/s p50 {h['c16_p50_ms']:.1f} p95 {h['c16_p95_ms']:.1f} | "
                  f"c1 p50 {h['c1_p50_ms']:.1f} p95 {h['c1_p95_ms']:.1f} ms | nDCG@10 {h['ndcg@10']:.4f}")
    print("\nsimulation (measured block + capacity) vs measured c=1 hit_frac, cold / warm:")
    for e in sim:
        m = e["measured"]
        print(f"  {e['setting']:7s} {e['dataset']:8s} {e['trace']} | sim {e['sim_cold']['hit_frac']:.3f} / {e['sim_warm']['hit_frac']:.3f} "
              f"| meas {m['c1_cold']['hit_frac']:.3f} / {m['c1_warm']['hit_frac']:.3f} | unbounded {e['sim_unbounded']['hit_frac']:.3f}")
    print("\nc=1 latency = a + b x (uncached tokens / 1000):")
    for s, f in latency_fit.items():
        print(f"  {s:7s} a {f['intercept_ms']:.2f} ms, b {f['ms_per_1k_uncached_tokens']:.2f} ms/1k tok, r {f['pearson_r']:.2f} "
              f"(n {f['n_requests']})")
    print("\ngates:")
    for s, g in gates.items():
        g2 = g["G2"]
        g3 = g.get("G3")
        print(f"  {s:7s} G1 {g['G1_pass']} | G2 ratio {g2['ratio']:.2f}{' FLAG' if g2['flag'] else ''}"
              + (f" | G3 literal {g3['literal_rule_16_pass']} at block {g3['actual_block_size']}: pass {g3['pass_at_actual_block']} exact {g3['exact_at_actual_block']}" if g3 else ""))
    print("\nwrote results/metrics/vllm.json, results/vllm/table3.csv")


if __name__ == "__main__":
    main()