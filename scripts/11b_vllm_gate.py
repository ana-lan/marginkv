# scripts/11b_vllm_gate.py
"""Phase 8a: vLLM gate check against a running server (rules pre-registered in results/vllm/gate_reference.json).

G1 yes/no logprobs present on every request (blocking)
G2 score agreement vs CPU fp32 on the 256-pair subset, ratio vs HF MPS bf16 (diagnostic, flag > 2.0)
G3 doc-first prefix-cache probes: same doc, two queries, cache reset per probe (blocking, df settings)
G4 is checked by 11_vllm_bench.py on the first cold trace 3 run.

Usage: python scripts/11b_vllm_gate.py --setting df --server-log artifacts/logs/serve_df.log
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import asyncio

import numpy as np

from marginkv.config import load_config
from marginkv.serving import client as sc
from marginkv.utils.io import ensure_dir, load_json, load_pt, run_info, save_json

BLOCK = 16
LOG_PATTERNS = ("GPU KV cache size", "Maximum concurrency", "backend", "kv_cache_dtype", "kv cache dtype",
                "fp8", "k_scale", "v_scale", "scale", "block_size", "prefix caching", "Prefix caching")


def errs(a, b):
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    return {"mean": float(d.mean()), "p95": float(np.percentile(d, 95)), "max": float(d.max()), "n": int(d.size)}


def flip_rate(a, b, group):
    n = f = 0
    for s in range(0, len(a), group):
        A, B = a[s:s + group], b[s:s + group]
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                n += 1
                f += (A[i] - A[j]) * (B[i] - B[j]) < 0
    return f / n


def tie_groups(a, group):
    return sum(len(set(a[s:s + group])) < len(a[s:s + group]) for s in range(0, len(a), group))


def lcp(x, y):
    n = 0
    for p, q in zip(x, y):
        if p != q:
            break
        n += 1
    return n


def grep_log(path, max_lines=120):
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            if any(p in line for p in LOG_PATTERNS):
                out.append(line.rstrip()[:400])
                if len(out) >= max_lines:
                    break
    return out


async def amain(args):
    cfg = load_config()
    res, arts = cfg.paths.results_dir, cfg.paths.artifacts_dir / "traces"
    ref = load_json(res / "vllm" / "gate_reference.json")
    sub = load_pt(arts / "gate_subset.pt")
    assert sub["subset_sha256"] == ref["subset_sha256"]
    datasets = list(ref["subset"].keys())
    order = "qf" if args.setting == "qf" else "df"
    hf_key = "qf_base" if order == "qf" else "df_lora_full"
    group = ref["settings"]["n_docs"]
    base = f"http://127.0.0.1:{args.port}"

    async with sc.new_session(16) as s:
        ready_s = await sc.wait_ready(s, base)
        info = await sc.server_info(s, base)
        sc.check_setting(info, args.setting)
        print(f"server ready ({ready_s:.0f} s): vLLM {info['version']} root {info['model_root']} cache {info['cache_config']}")

        # G1 + G2 on the 256-pair subset
        prompts = [p for ds in datasets for p in sub["datasets"][ds][f"{order}_ids"]]
        await sc.reset_prefix_cache(s, base)
        out, wall = await sc.run_requests(s, base, prompts, 16, "marginkv", args.logprobs, cfg.seed)
        lo = [sc.log_odds(r) for r in out]
        missing = sum(x is None for x in lo)
        errors = [r["error"] for r in out if r["error"]]
        g1 = {"n": len(lo), "missing": missing, "errors": len(errors), "first_error": errors[0] if errors else None,
              "pass": missing == 0 and not errors}

        fp32 = [x for ds in datasets for x in ref["fp32"][order][ds]]
        hf = [x for ds in datasets for x in ref["hf_bf16"][ds][hf_key]]
        g2 = None
        if missing == 0:
            ev, eh = errs(lo, fp32), errs(hf, fp32)
            g2 = {"vllm_vs_fp32": ev, "hf_bf16_vs_fp32": eh, "ratio": ev["mean"] / eh["mean"],
                  "flag_ratio_gt_2": ev["mean"] / eh["mean"] > 2.0,
                  "pair_flip_rate_vs_fp32": {"vllm": flip_rate(lo, fp32, group), "hf_bf16": flip_rate(hf, fp32, group)},
                  "tie_groups": {"vllm": tie_groups(lo, group), "hf_bf16": tie_groups(hf, group),
                                 "groups": len(lo) // group},
                  "per_dataset": {}}
            i = 0
            for ds in datasets:
                n = len(ref["subset"][ds])
                g2["per_dataset"][ds] = {"vllm_vs_fp32": errs(lo[i:i + n], fp32[i:i + n]),
                                         "hf_bf16_vs_fp32": errs(hf[i:i + n], fp32[i:i + n])}
                i += n

        # G3 prefix-cache probes (doc-first only)
        g3 = None
        if order == "df":
            probes = []
            for ds in datasets:
                tr = load_pt(arts / f"{ds}_t2.pt")
                pairs = tr["pairs"]
                starts = [i for i in range(len(pairs) - 1)
                          if (i == 0 or pairs[i][1] != pairs[i - 1][1]) and pairs[i + 1][1] == pairs[i][1]][:args.probes]
                for st in starts:
                    a_ids, b_ids, cl = tr["df_ids"][st], tr["df_ids"][st + 1], tr["df_cache_len"][st]
                    await sc.reset_prefix_cache(s, base)
                    m0 = await sc.counters(s, base)
                    ra = await sc.complete(s, base, a_ids, "marginkv", args.logprobs, cfg.seed)
                    m1 = await sc.counters(s, base)
                    rb = await sc.complete(s, base, b_ids, "marginkv", args.logprobs, cfg.seed)
                    m2 = await sc.counters(s, base)
                    a_hit = ra["cached"] if ra["cached"] is not None else m1["hits"] - m0["hits"]
                    b_hit = rb["cached"] if rb["cached"] is not None else m2["hits"] - m1["hits"]
                    min_expected = (cl // BLOCK) * BLOCK
                    sim_expected = min(lcp(a_ids, b_ids) // BLOCK, (len(b_ids) - 1) // BLOCK) * BLOCK
                    probes.append({"dataset": ds, "doc": pairs[st][1], "queries": [pairs[st][0], pairs[st + 1][0]],
                                   "cache_len": cl, "len_b": len(b_ids),
                                   "a_cached": a_hit, "b_cached": b_hit,
                                   "a_metric_hits": m1["hits"] - m0["hits"], "b_metric_hits": m2["hits"] - m1["hits"],
                                   "usage_cached_reported": ra["cached"] is not None,
                                   "min_expected": min_expected, "sim_expected": sim_expected})
            g3 = {"probes": probes,
                  "pass": bool(probes) and all(p["a_cached"] == 0 and p["b_cached"] >= p["min_expected"] for p in probes),
                  "matches_simulator_rule": all(p["b_cached"] == p["sim_expected"] for p in probes),
                  "usage_cached_reported": all(p["usage_cached_reported"] for p in probes)}

    blocking = g1["pass"] and (g3 is None or g3["pass"])
    out_obj = {"run_info": run_info(), "phase": "8a", "setting": args.setting, "server": info,
               "logprobs": args.logprobs, "logprobs_mode": args.logprobs_mode,
               "rules": {k: ref["rules"][k] for k in ("G1_yes_no_path", "G2_score_agreement", "G3_prefix_cache")},
               "G1": g1, "G2": g2, "G3": g3, "blocking_pass": blocking,
               "server_log_lines": grep_log(args.server_log)}
    path = ensure_dir(res / "vllm") / f"gate_{args.setting}.json"
    save_json(out_obj, path)

    print(f"G1 yes/no present: {'PASS' if g1['pass'] else 'FAIL'} (missing {missing}/{len(lo)}, errors {len(errors)})")
    if errors:
        print("  first error:", errors[0])
    if g2:
        print(f"G2 mean |d| vs fp32: vLLM {g2['vllm_vs_fp32']['mean']:.4f} HF bf16 {g2['hf_bf16_vs_fp32']['mean']:.4f} "
              f"ratio {g2['ratio']:.2f}{'  FLAG' if g2['flag_ratio_gt_2'] else ''} | flips {g2['pair_flip_rate_vs_fp32']} "
              f"| tie groups {g2['tie_groups']}")
    if g3:
        print(f"G3 prefix cache: {'PASS' if g3['pass'] else 'FAIL'} | simulator rule exact: {g3['matches_simulator_rule']} "
              f"| usage cached reported: {g3['usage_cached_reported']}")
        for p in g3["probes"][:4]:
            print(f"   {p['dataset']} doc {p['doc']}: cache_len {p['cache_len']} a {p['a_cached']} b {p['b_cached']} "
                  f"(min {p['min_expected']}, sim {p['sim_expected']})")
    print(f"blocking checks: {'PASS' if blocking else 'FAIL'} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, choices=["qf", "df", "df_fp8"])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--logprobs", type=int, default=2)
    ap.add_argument("--logprobs-mode", default="processed_logprobs", help="record only; must match the server flag")
    ap.add_argument("--probes", type=int, default=5, help="G3 probes per dataset")
    ap.add_argument("--server-log", default=None)
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()