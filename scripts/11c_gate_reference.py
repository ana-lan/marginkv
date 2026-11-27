# scripts/11c_gate_reference.py
"""Phase 8a.2: CPU fp32 reference scores for the vLLM gate subset, HF baselines for G4,
and the pre-registered Phase 8a rules. Commit the output before any vLLM run.

Subset = tier 2a's: first 16 queries x top 8 docs per dataset (256 pairs pooled).
  query-first prompts on the original Qwen3-Reranker-0.6B (what the qf setting serves)
  doc-first variant B prompts on the merged doc-first LoRA (what the df settings serve)

Outputs:
  results/vllm/gate_reference.json        rules, subset, fp32 + HF bf16 scores, errors, G4 baselines
  artifacts/traces/gate_subset.pt         token ids for the pod
  artifacts/traces/qrels_{ds}.json        qrels for on-pod nDCG (data/ is not shipped)
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import gc
import time

import numpy as np
import torch

from marginkv.config import load_config
from marginkv.data.beir_loader import load_beir
from marginkv.eval.rerank import evaluate_rerank
from marginkv.models import qwen3_reranker as qr
from marginkv.train.data import content_sha256, sha256_file
from marginkv.utils.device import get_dtype
from marginkv.utils.io import ensure_dir, load_json, load_pt, run_info, save_json, save_pt
from marginkv.utils.seed import set_seed

RULES = {
    "registered": "2026-11-26, before any vLLM run (Phase 8a.2)",
    "settings": {
        "qf": "original Qwen3-Reranker-0.6B @ pinned revision, query-first prompts, BF16 KV, prefix caching on",
        "df": "merged doc-first LoRA (artifacts/lora/df/merged, sha256 checked), variant B doc-first prompts, BF16 KV, prefix caching on",
        "df_fp8": "same as df with --kv-cache-dtype fp8 (vLLM native; record its scale handling, do not equate with our per-head scales)",
        "shared": "identical server flags otherwise (max_model_len 512, gpu_memory_utilization 0.90, seed = cfg.seed); "
                  "token-id prompts only; max_tokens 1; temperature 0; allowed_token_ids [no, yes]; "
                  "score = logprob(yes) - logprob(no); vLLM version pinned in requirements-vllm.txt",
    },
    "G1_yes_no_path": "blocking: every gate request returns logprobs for both yes (9693) and no (2152); pass = 100% of requests",
    "G2_score_agreement": "diagnostic, not blocking: per order on the 256-pair subset, mean |vLLM bf16 - HF fp32| vs "
                          "mean |HF MPS bf16 - HF fp32| (full prompt); report ratio, max, pair flip rate, tie count; "
                          "flag if ratio > 2.0 (vLLM computes yes/no logits in bf16, ours in fp32)",
    "G3_prefix_cache": "blocking: on doc-first probes (same doc, two different queries, prefix cache reset before each probe), "
                       "the second request's reported cached tokens >= floor(cache_len / 16) * 16 (block-aligned doc prefix) "
                       "on every probe; first request after reset reports 0. Exact agreement with the 8a.1 simulator rule "
                       "is recorded separately (validates the simulation, not the gate)",
    "G4_quality": "blocking for df: vLLM df (BF16 KV) trace 3, first cold run, nDCG@10 within 0.01 of HF doc-first LoRA "
                  "full prompt (|delta| <= 0.01) on both datasets; same check reported for qf vs the original model "
                  "(qf failing marks qf quality only; its systems numbers are still reported). Paired CI and W/L/T reported",
    "fp8_vs_bf16_in_vllm": "reported: df_fp8 vs df on the same trace 3 pairs, paired.compare + paired.verdict with the "
                           "5.2 margins (results/metrics/quant_noise_floor.json)",
    "measurement": {
        "cold": "prefix cache reset immediately before the trace",
        "warm": "the same trace rerun immediately after the cold run, no reset",
        "concurrency": [1, 4, 16],
        "latency": "client-side end to end per request (send to response), p50 / p95",
        "throughput": "requests/s and prompt tokens/s over the trace's wall time",
        "hit_tokens": "per request usage.prompt_tokens_details.cached_tokens; doc-token reuse = cached tokens inside "
                      "each request's doc span / doc tokens (same definition as the 8a.1 simulation)",
        "recorded": "physical block size, num_gpu_blocks, prefix-match granularity, attention backend, kv cache dtype, fp8 scale handling",
    },
    "claims": "only measured numbers go in results and README; trace 3 reported as measured, including little benefit; "
              "no vLLM throughput claims for INT4 or pruned caches",
    "pod_cut_order": ["gate (G1, G3 on smoke; G2 subset) <= 60 min",
                      "trace 3, c=16, cold + warm, settings df, qf, df_fp8 (G4 from the first df run)",
                      "trace 3, c=1 and c=4, cold + warm",
                      "traces 1 and 2, all settings, c=16 then c=1, 4"],
    "fallback": "if G1, G3 or df G4 is not passing by 12:00 CT Fri Nov 27, or no pod is available, apply with the "
                "Phase 5 resume; apply by 18:00 CT regardless",
}


def subset_pairs(art, n_queries, n_docs):
    return [[q, d] for q in art["query_order"][:n_queries] for d in art["rerank_candidates"][q][:n_docs]]


def errors(a, b):
    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return {"mean": float(d.mean()), "max": float(d.max()), "p95": float(np.percentile(d, 95)), "n": int(d.size)}


def score_fp32(model_ref, seqs, batch_size, device):
    name, revision = model_ref
    model = qr.load_model(name, revision, device, get_dtype("float32"))
    t0 = time.time()
    out = qr.score_sequences(model, seqs, batch_size, device)
    secs = time.time() - t0
    del model
    gc.collect()
    return [float(x) for x in out["log_odds"]], secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--n-queries", type=int, default=16)
    ap.add_argument("--n-docs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    datasets = args.datasets or list(cfg.datasets)
    res, arts = cfg.paths.results_dir, ensure_dir(cfg.paths.artifacts_dir / "traces")
    depth = cfg.rerank.depth

    merged = cfg.paths.artifacts_dir / "lora" / "df" / "merged"
    files_sha = load_json(merged / "marginkv_merge.json")["files_sha256"]
    for fn, sha in files_sha.items():
        assert sha256_file(merged / fn) == sha, f"sha256 mismatch: {fn}"
    print(f"merged df LoRA: {len(files_sha)} files sha256 ok")

    margins = load_json(res / "metrics" / "quant_noise_floor.json")["margins"]

    subset, ids, hf_bf16, baselines = {}, {}, {}, {}
    for ds in datasets:
        tok = cfg.paths.artifacts_dir / "tokenized"
        art, dfart = load_pt(tok / f"{ds}_qwen3.pt"), load_pt(tok / f"{ds}_qwen3_df.pt")
        assert dfart["base_content_sha256"] == art["content_sha256"]
        pieces = {**art["pieces"], **dfart["seam_pieces"]}
        pairs = subset_pairs(art, args.n_queries, args.n_docs)
        subset[ds] = pairs
        ids[ds] = {
            "pairs": pairs,
            "qf_ids": [qr.query_first_ids(pieces, art["query_qf"][q], art["doc"][d]) for q, d in pairs],
            "df_ids": [qr.doc_first_seam_ids(pieces, dfart["doc_seam"][d], art["query_df"][q]) for q, d in pairs],
            "df_cache_len": [dfart["cache_len_seam"][d] for _, d in pairs],
        }

        files = {"qf_base": f"qwen3_{ds}.json", "df_lora_full": f"qwen3_lora_dfB_{ds}.json",
                 "df_lora_cached_p5_b8": f"qwen3_lora_dfB_cached_p5_b8_{ds}.json"}
        scores = {k: load_json(res / "scores" / v)["scores"] for k, v in files.items()}
        hf_bf16[ds] = {k: [float(scores[k][q][d]) for q, d in pairs] for k in scores}

        cands = load_json(res / "candidates" / f"{ds}.json")["candidates"]
        qrels = load_beir(ds, cfg.split, data_dir=str(cfg.paths.data_dir), download=False).qrels
        save_json(qrels, arts / f"qrels_{ds}.json")
        baselines[ds] = {}
        for k in files:
            ev = evaluate_rerank(cands, scores[k], qrels, depth)
            baselines[ds][k] = {"file": files[k], "mean": ev["mean"], "per_query": ev["per_query"]}
        print(f"{ds}: {len(pairs)} subset pairs | nDCG@10 qf_base {baselines[ds]['qf_base']['mean']['ndcg@10']:.4f} "
              f"df_lora_full {baselines[ds]['df_lora_full']['mean']['ndcg@10']:.4f} "
              f"df_cached {baselines[ds]['df_lora_cached_p5_b8']['mean']['ndcg@10']:.4f}")

    # fp32 references, one model at a time, pooled over datasets
    qf_all = [s for ds in datasets for s in ids[ds]["qf_ids"]]
    df_all = [s for ds in datasets for s in ids[ds]["df_ids"]]
    print(f"scoring {len(df_all)} doc-first pairs, CPU fp32 (merged df LoRA) ...")
    df32, t_df = score_fp32((str(merged), None), df_all, args.batch_size, device)
    print(f"  {t_df:.0f} s")
    print(f"scoring {len(qf_all)} query-first pairs, CPU fp32 (original model) ...")
    qf32, t_qf = score_fp32((cfg.models.qwen3.name, cfg.models.qwen3.revision), qf_all, args.batch_size, device)
    print(f"  {t_qf:.0f} s")

    fp32, i = {"qf": {}, "df": {}}, 0
    for ds in datasets:
        n = len(subset[ds])
        fp32["qf"][ds], fp32["df"][ds] = qf32[i:i + n], df32[i:i + n]
        i += n

    pooled = lambda d: [x for ds in datasets for x in d[ds]]
    err = {
        "qf_base_bf16_vs_fp32": {"pooled": errors(pooled({ds: hf_bf16[ds]["qf_base"] for ds in datasets}), pooled(fp32["qf"])),
                                 **{ds: errors(hf_bf16[ds]["qf_base"], fp32["qf"][ds]) for ds in datasets}},
        "df_full_bf16_vs_fp32": {"pooled": errors(pooled({ds: hf_bf16[ds]["df_lora_full"] for ds in datasets}), pooled(fp32["df"])),
                                 **{ds: errors(hf_bf16[ds]["df_lora_full"], fp32["df"][ds]) for ds in datasets}},
        "df_cached_bf16_vs_fp32": {"pooled": errors(pooled({ds: hf_bf16[ds]["df_lora_cached_p5_b8"] for ds in datasets}), pooled(fp32["df"])),
                                   **{ds: errors(hf_bf16[ds]["df_lora_cached_p5_b8"], fp32["df"][ds]) for ds in datasets}},
    }
    tier2a = load_json(res / "metrics" / "equality_lora.json")["tier2a"]
    err["tier2a_check"] = {"tier2a_full": tier2a["mean_err_full_bf16"], "tier2a_cached": tier2a["mean_err_cached_bf16"],
                           "recomputed_full": err["df_full_bf16_vs_fp32"]["pooled"]["mean"],
                           "recomputed_cached": err["df_cached_bf16_vs_fp32"]["pooled"]["mean"]}

    out = {"run_info": run_info(), "phase": "8a.2", "rules": RULES, "margins": margins,
           "settings": {"n_queries": args.n_queries, "n_docs": args.n_docs, "batch_size": args.batch_size,
                        "device": "cpu", "dtype": "float32", "depth": depth},
           "models": {"qf": {"name": cfg.models.qwen3.name, "revision": cfg.models.qwen3.revision},
                      "df": {"dir": "artifacts/lora/df/merged", "files_sha256": files_sha}},
           "subset": subset, "subset_sha256": content_sha256(subset),
           "fp32": fp32, "hf_bf16": hf_bf16, "errors": err,
           "timing_s": {"df_fp32": t_df, "qf_fp32": t_qf},
           "baselines": baselines}
    save_json(out, ensure_dir(res / "vllm") / "gate_reference.json")
    save_pt({"subset_sha256": out["subset_sha256"], "datasets": ids}, arts / "gate_subset.pt")

    print("\nmean |bf16 - fp32| log-odds (pooled):")
    for k in ("qf_base_bf16_vs_fp32", "df_full_bf16_vs_fp32", "df_cached_bf16_vs_fp32"):
        e = err[k]["pooled"]
        print(f"  {k:26s} mean {e['mean']:.4f}  p95 {e['p95']:.4f}  max {e['max']:.4f}")
    print(f"  tier2a check: full {tier2a['mean_err_full_bf16']:.4f} vs {err['tier2a_check']['recomputed_full']:.4f}, "
          f"cached {tier2a['mean_err_cached_bf16']:.4f} vs {err['tier2a_check']['recomputed_cached']:.4f}")
    print("wrote results/vllm/gate_reference.json, artifacts/traces/gate_subset.pt")


if __name__ == "__main__":
    main()