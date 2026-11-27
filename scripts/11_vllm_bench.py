# scripts/11_vllm_bench.py
"""Phase 8a: run traces against a running vLLM server (one setting per server).

Per (dataset, trace, concurrency): cold run (prefix cache reset first) then warm run (same trace, no reset).
Records per-request latency, cached tokens, scores; Prometheus prefix-cache counter deltas; throughput.
On trace 3 cold runs: G4 quality vs the HF baseline, and for df_fp8 the paired verdict vs vLLM df (same concurrency).

Usage: python scripts/11_vllm_bench.py --setting df --traces t3 --concurrency 16
Outputs: results/vllm/runs/{setting}_{trace}_{ds}_c{N}_{cold|warm}.json  (--smoke N: artifacts/smoke/vllm/)
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import asyncio

from marginkv.config import load_config
from marginkv.eval.paired import compare, score_agreement, verdict
from marginkv.eval.rerank import evaluate_rerank
from marginkv.serving import client as sc
from marginkv.utils.io import ensure_dir, load_json, load_pt, run_info, save_json

ORDER = {"qf": "qf", "df": "df", "df_fp8": "df"}
REF_KEY = {"qf": "qf_base", "df": "df_lora_full", "df_fp8": "df_lora_full"}
MODES = ("cold", "warm")


def nested_scores(pairs, res):
    out, missing = {}, 0
    for (q, d), r in zip(pairs, res):
        lo = sc.log_odds(r)
        if lo is None:
            missing += 1
            lo = -1e9
        out.setdefault(q, {})[d] = lo
    return out, missing


def quality(cfg, ds, scores):
    cands = load_json(cfg.paths.results_dir / "candidates" / f"{ds}.json")["candidates"]
    qrels = load_json(cfg.paths.artifacts_dir / "traces" / f"qrels_{ds}.json")
    return cands, evaluate_rerank(cands, scores, qrels, cfg.rerank.depth)


async def run_pair(s, base, args, cfg, info, setting, ds, trace, c, tr, n, out_dir, ref):
    order = ORDER[setting]
    prompts, spans, pairs = tr[f"{order}_ids"][:n], tr[f"{order}_span"][:n], tr["pairs"][:n]
    lengths = [len(p) for p in prompts]
    full = n == len(tr["pairs"])
    for mode in MODES:
        if mode == "cold":
            await sc.reset_prefix_cache(s, base)
        m0 = await sc.counters(s, base)
        res, wall = await sc.run_requests(s, base, prompts, c, "marginkv", args.logprobs, cfg.seed)
        m1 = await sc.counters(s, base)
        summ = sc.summarize(res, lengths, spans, wall)
        dq, dh = m1["queries"] - m0["queries"], m1["hits"] - m0["hits"]
        summ["metrics_delta"] = {"queries": dq, "hits": dh, "hit_frac": dh / dq if dq else None}
        scores, missing = nested_scores(pairs, res)

        out = {"run_info": run_info(), "phase": "8a", "setting": setting, "dataset": ds, "trace": trace,
               "concurrency": c, "mode": mode, "smoke": None if full else n, "server": info,
               "logprobs": args.logprobs, "logprobs_mode": args.logprobs_mode,
               "pairs_sha256": tr["pairs_sha256"], "summary": summ,
               "per_request": {"latency_ms": [round(r["latency_ms"], 3) for r in res],
                               "cached": [r["cached"] for r in res],
                               "log_odds": [sc.log_odds(r) for r in res]},
               "scores": scores if trace == "t3" else None}

        if trace == "t3" and full and mode == "cold" and missing == 0:
            cands, ev = quality(cfg, ds, scores)
            b = ref["baselines"][ds][REF_KEY[setting]]
            qids = list(b["per_query"].keys())
            cmp = compare(b["per_query"], ev["per_query"], qids, cfg.seed)
            q = {"mean": ev["mean"], "diagnostics": ev["diagnostics"], "baseline": REF_KEY[setting],
                 "baseline_mean": b["mean"], "vs_baseline": cmp,
                 "G4_pass": abs(cmp["delta_ndcg@10"]) <= 0.01}
            if setting == "df_fp8":
                dfp = ensure_dir(out_dir) / f"df_t3_{ds}_c{c}_cold.json"
                if dfp.exists():
                    df_scores = load_json(dfp)["scores"]
                    _, ev_df = quality(cfg, ds, df_scores)
                    c2 = compare(ev_df["per_query"], ev["per_query"], qids, cfg.seed)
                    margin = ref["margins"][ds]
                    q["vs_vllm_df_bf16"] = {**c2, "margin": margin,
                                            "verdict": verdict(c2["delta_ndcg@10"], c2["ndcg@10_ci95"], margin),
                                            "scores": score_agreement(cands, df_scores, scores, qids, cfg.rerank.depth)}
            out["quality"] = q

        path = ensure_dir(out_dir) / f"{setting}_{trace}_{ds}_c{c}_{mode}.json"
        save_json(out, path)
        hit = summ.get("hit_frac")
        doc = summ.get("doc_tokens_reused_frac")
        line = (f"[{setting} {trace} {ds} c{c} {mode}] {summ['n']} req {summ['req_per_s']:.1f} req/s "
                f"{summ['prompt_tok_per_s']:.0f} tok/s p50 {summ['latency_ms']['p50']:.1f} p95 {summ['latency_ms']['p95']:.1f} ms | "
                f"hit {hit if hit is None else round(hit, 3)} doc {doc if doc is None else round(doc, 3)} "
                f"metric hit {summ['metrics_delta']['hit_frac'] and round(summ['metrics_delta']['hit_frac'], 3)} | missing {missing}")
        print(line, flush=True)
        if "quality" in out:
            qq = out["quality"]
            print(f"    nDCG@10 {qq['mean']['ndcg@10']:.4f} vs {qq['baseline']} {qq['baseline_mean']['ndcg@10']:.4f} "
                  f"d {qq['vs_baseline']['delta_ndcg@10']:+.4f} {qq['vs_baseline']['ndcg@10_ci95']} "
                  f"G4 {'PASS' if qq['G4_pass'] else 'FAIL'}", flush=True)
            if "vs_vllm_df_bf16" in qq:
                v = qq["vs_vllm_df_bf16"]
                print(f"    fp8 vs vLLM bf16: d {v['delta_ndcg@10']:+.4f} {v['ndcg@10_ci95']} -> {v['verdict']}", flush=True)


async def amain(args):
    cfg = load_config()
    ref = load_json(cfg.paths.results_dir / "vllm" / "gate_reference.json")
    datasets = args.datasets or list(cfg.datasets)
    base = f"http://127.0.0.1:{args.port}"
    out_dir = (cfg.paths.artifacts_dir / "smoke" / "vllm") if args.smoke else (cfg.paths.results_dir / "vllm" / "runs")

    async with sc.new_session(max(args.concurrency)) as s:
        await sc.wait_ready(s, base)
        info = await sc.server_info(s, base)
        sc.check_setting(info, args.setting)
        print(f"vLLM {info['version']} | {info['model_root']} | cache {info['cache_config']}", flush=True)

        warmed = False
        for ds in datasets:
            for trace in args.traces:
                tr = load_pt(cfg.paths.artifacts_dir / "traces" / f"{ds}_{trace}.pt")
                n = min(args.smoke, len(tr["pairs"])) if args.smoke else len(tr["pairs"])
                if not warmed:
                    await sc.run_requests(s, base, tr[f"{ORDER[args.setting]}_ids"][:32], 16, "marginkv",
                                          args.logprobs, cfg.seed)
                    warmed = True
                for c in args.concurrency:
                    paths = [out_dir / f"{args.setting}_{trace}_{ds}_c{c}_{m}.json" for m in MODES]
                    if all(p.exists() for p in paths) and not args.force:
                        print(f"skip {args.setting} {trace} {ds} c{c} (exists)", flush=True)
                        continue
                    await run_pair(s, base, args, cfg, info, args.setting, ds, trace, c, tr, n, out_dir, ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, choices=["qf", "df", "df_fp8"])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--traces", nargs="+", default=["t3"], choices=["t1", "t2", "t3"])
    ap.add_argument("--concurrency", nargs="+", type=int, default=[16])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--logprobs", type=int, default=2)
    ap.add_argument("--logprobs-mode", default="processed_logprobs", help="record only; must match the server flag")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()