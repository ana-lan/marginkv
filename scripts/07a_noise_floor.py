# scripts/07a_noise_floor.py
"""Phase 5.2: noise floor of the BF16 cached path and the pre-registered equivalence margin.

Model: merged doc-first LoRA (artifacts/lora/df/merged), variant B, cached path, MPS bf16.
Each doc KV is built once (build batch 1) and every top-20 pair is scored at score batch 8 and 16
on the same KV. The runs differ only in batch composition, a bf16 perturbation with no relevance
information, so their difference is the yardstick for "no measurable loss" in Phase 5.4.

PRE-REGISTERED (5.2, before any noise-floor or compressed data):
  m_ds = max(0.002, max(|lo|, |hi|) of the paired 95% bootstrap CI of nDCG@10(b16) - nDCG@10(b8))
  5.4 verdicts per format and dataset, vs BF16 cached at score batch 8 (this script's b8 run):
    no_measurable_loss: 95% CI of delta nDCG@10 inside [-m_ds, +m_ds]
    within_1_point:     point delta nDCG@10 >= -0.01
    loss:               otherwise (report the number)
  Flag (no rule change) if any m_ds > 0.005.
The b8 run is the Phase 5 BF16 baseline. It is also checked against the frozen Phase 4 cached file.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse

import numpy as np
import torch

from marginkv.config import load_config
from marginkv.data.beir_loader import load_beir
from marginkv.eval.rerank import evaluate_rerank, rerank_merge
from marginkv.kv import sweep
from marginkv.models import qwen3_reranker as qr
from marginkv.train.data import content_sha256, sha256_file
from marginkv.utils.device import get_device, get_dtype
from marginkv.utils.io import ensure_dir, load_json, load_pt, run_info, save_json, save_pt
from marginkv.utils.seed import set_seed

BATCHES = (8, 16)
MARGIN_FLOOR = 0.002
MARGIN_FLAG = 0.005
N_BOOT = 10000
RULES = {
    "margin": "m_ds = max(0.002, max(|lo|,|hi|) of paired 95% bootstrap CI of nDCG@10(b16) - nDCG@10(b8))",
    "no_measurable_loss": "95% CI of delta nDCG@10 (format - bf16 b8) inside [-m_ds, +m_ds]",
    "within_1_point": "point delta nDCG@10 >= -0.01",
    "loss": "otherwise",
    "flag": "any m_ds > 0.005 -> noise floor too coarse to call small losses (flag only)",
    "bootstrap": f"percentile, {N_BOOT} paired query resamples, seed = cfg.seed",
}


def verify_model(model_dir):
    shas = load_json(model_dir / "marginkv_merge.json")["files_sha256"]
    for fname, want in shas.items():
        if sha256_file(model_dir / fname) != want:
            raise SystemExit(f"sha mismatch: {fname}")
    return shas


def paired_ci(a, b, seed):
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), [float(lo), float(hi)]


def wlt(a, b, eps=1e-9):
    w = sum(1 for x, y in zip(a, b) if y > x + eps)
    l = sum(1 for x, y in zip(a, b) if y < x - eps)
    return [w, l, len(a) - w - l]


def agreement(cands, sa, sb, qids, depth):
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


def score_record(meta, state, name, sb, qids, art):
    out = state["out"][name]
    nest = lambda k: {q: {d: round(out[q][d][k], 6) for d in art["rerank_candidates"][q]} for q in qids}
    n_pairs = sum(len(out[q]) for q in qids)
    return {**meta, "settings": {**meta["settings"], "score_batch": sb},
            "num_pairs": n_pairs, "score_key": "log_odds", "aux_key": "p_yes",
            "scores": nest(0), "logit_yes": nest(1), "logit_no": nest(2), "aux": nest(3),
            "timing": {"build_total_s": sum(state["build_ms"]) / 1000,
                       "build_ms_per_doc": float(np.mean(state["build_ms"])),
                       "score_total_s": sum(state["score_ms"][name]) / 1000,
                       "score_ms_per_pair": sum(state["score_ms"][name]) / n_pairs}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="first N queries per dataset, outputs under artifacts/smoke")
    ap.add_argument("--doc-chunk", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)
    torch.set_grad_enabled(False)
    device, dtype = get_device(cfg.runtime.device), get_dtype(cfg.runtime.dtype)
    A, R = cfg.paths.artifacts_dir, cfg.paths.results_dir
    smoke = args.smoke > 0
    score_dir = ensure_dir(A / "smoke" / "scores" if smoke else R / "scores")
    part_dir = ensure_dir(A / "smoke_partial" if smoke else A / "scores_partial")
    metrics_path = (A / "smoke" / "metrics" if smoke else R / "metrics") / "quant_noise_floor.json"
    depth = cfg.rerank.depth

    model_dir = A / "lora" / "df" / "merged"
    shas = verify_model(model_dir)
    model = None   # loaded lazily (skipped if all score files exist)

    report = {"run_info": run_info(), "phase": "5.2", "rules": RULES, "batches": list(BATCHES),
              "smoke_queries": args.smoke, "datasets": {}}

    for ds in cfg.datasets:
        art = load_pt(A / "tokenized" / f"{ds}_qwen3.pt")
        dfart = load_pt(A / "tokenized" / f"{ds}_qwen3_df.pt")
        assert dfart["base_content_sha256"] == art["content_sha256"], "seam artifact mismatch"
        qids = list(art["query_order"])[:args.smoke] if smoke else list(art["query_order"])
        cands_all = load_json(R / "candidates" / f"{ds}.json")["candidates"]
        cands = {q: cands_all[q] for q in qids}
        beir = load_beir(ds, cfg.split, data_dir=str(cfg.paths.data_dir), download=False)
        qrels = {q: beir.qrels[q] for q in qids}

        inputs_sha = content_sha256({"art": art["content_sha256"], "dfart": dfart["content_sha256"], "qids": qids})
        guard = {"inputs_sha256": inputs_sha, "model_files_sha256": shas, "doc_chunk": args.doc_chunk,
                 "batches": list(BATCHES), "device": str(device), "dtype": cfg.runtime.dtype}
        paths = {b: score_dir / f"qwen3_lora_dfB_cached_p5_b{b}_{ds}.json" for b in BATCHES}
        partial = part_dir / f"noise_{ds}.pt"

        have = all(p.exists() for p in paths.values()) and not args.force
        if have:
            for b, p in paths.items():
                if load_json(p)["inputs_sha256"] != inputs_sha:
                    raise SystemExit(f"{p.name} was made from different inputs; rerun with --force")
            print(f"[{ds}] score files exist, reusing")
        else:
            if model is None:
                model = qr.load_model(str(model_dir), None, device, dtype)
            state = None
            if partial.exists() and not args.force:
                p = load_pt(partial)
                if p["guard"] != guard:
                    raise SystemExit(f"{partial} has different settings; rerun with --force")
                state = p["state"]
                print(f"[{ds}] resuming at chunk {state['done_chunks']}")

            def on_chunk(st, done, total):
                save_pt({"guard": guard, "state": st}, partial)
                if done % 20 == 0 or done == total:
                    sm = {n: f"{np.mean(v):.0f}" for n, v in st["score_ms"].items()}
                    print(f"[{ds}] chunk {done}/{total}  build {np.mean(st['build_ms']):.0f} ms/doc  "
                          f"score ms/batch {sm}", flush=True)

            configs = {f"b{b}": (sweep.identity, b) for b in BATCHES}
            state = sweep.run_sweep(model, art, dfart, device, configs, qids=qids,
                                    doc_chunk=args.doc_chunk, state=state, on_chunk=on_chunk)
            meta = {"run_info": run_info(), "phase": "5.2", "model_dir": str(model_dir), "model_files_sha256": shas,
                    "dataset": ds, "split": cfg.split, "variant": "B", "prompt_order": "doc_first", "path": "cached",
                    "device": str(device), "dtype": cfg.runtime.dtype, "depth": depth,
                    "settings": {"doc_chunk": args.doc_chunk, "build_batch": 1},
                    "inputs_sha256": inputs_sha, "smoke_queries": args.smoke,
                    "num_docs": state["done_chunks"] and len(state["build_ms"]), "query_order": qids}
            for b in BATCHES:
                save_json(score_record(meta, state, f"b{b}", b, qids, art), paths[b])
            partial.unlink(missing_ok=True)

        scores = {b: load_json(paths[b])["scores"] for b in BATCHES}
        ev = {b: evaluate_rerank(cands, scores[b], qrels, depth) for b in BATCHES}
        pq = {b: ev[b]["per_query"] for b in BATCHES}
        n8, n16 = [pq[8][q]["ndcg@10"] for q in qids], [pq[16][q]["ndcg@10"] for q in qids]
        m8, m16 = [pq[8][q]["mrr@10"] for q in qids], [pq[16][q]["mrr@10"] for q in qids]
        d_n, ci_n = paired_ci(n8, n16, cfg.seed)
        d_m, ci_m = paired_ci(m8, m16, cfg.seed)
        margin = max(MARGIN_FLOOR, abs(ci_n[0]), abs(ci_n[1]))

        frozen = None
        fz_path = R / "scores" / f"qwen3_lora_dfB_cached_{ds}.json"
        if not smoke and fz_path.exists():
            fz = load_json(fz_path)["scores"]
            frozen = {}
            for b in BATCHES:
                diffs = [abs(scores[b][q][d] - fz[q][d]) for q in qids for d in scores[b][q]]
                frozen[f"b{b}_vs_frozen"] = {"max_abs": max(diffs),
                                             "identical_frac": sum(x <= 1e-6 for x in diffs) / len(diffs)}
            fz_ev = evaluate_rerank(cands, fz, qrels, depth)
            frozen["frozen_ndcg@10"] = fz_ev["mean"]["ndcg@10"]

        r = {"num_queries": len(qids),
             "b8": {"mean": ev[8]["mean"], "diagnostics": ev[8]["diagnostics"]},
             "b16": {"mean": ev[16]["mean"], "diagnostics": ev[16]["diagnostics"]},
             "noise": {"delta_ndcg@10": d_n, "ndcg@10_ci95": ci_n, "delta_mrr@10": d_m, "mrr@10_ci95": ci_m,
                       "wlt_ndcg@10": wlt(n8, n16), "scores": agreement(cands, scores[8], scores[16], qids, depth)},
             "margin": margin, "margin_flag": margin > MARGIN_FLAG, "frozen_check": frozen}
        report["datasets"][ds] = r
        a = r["noise"]["scores"]
        print(f"[{ds}] b8 nDCG {ev[8]['mean']['ndcg@10']:.4f} MRR {ev[8]['mean']['mrr@10']:.4f} | "
              f"b16 - b8: dNDCG {d_n:+.4f} [{ci_n[0]:+.4f}, {ci_n[1]:+.4f}] dMRR {d_m:+.4f} "
              f"W/L/T {r['noise']['wlt_ndcg@10']} | scores max {a['max_abs']:.3f} p99 {a['p99_abs']:.3f} "
              f"flips {a['pair_flip_rate']:.1e} same-order {a['identical_topk_order']} | margin {margin:.4f}"
              f"{'  FLAG' if r['margin_flag'] else ''}")
        if frozen:
            print(f"[{ds}] vs frozen Phase 4 cached: b8 max {frozen['b8_vs_frozen']['max_abs']:.2e} "
                  f"identical {frozen['b8_vs_frozen']['identical_frac']:.3f}, "
                  f"b16 max {frozen['b16_vs_frozen']['max_abs']:.2e}; frozen nDCG {frozen['frozen_ndcg@10']:.4f}")

    report["margins"] = {ds: r["margin"] for ds, r in report["datasets"].items()}
    save_json(report, metrics_path)
    print(f"margins: {report['margins']}")


if __name__ == "__main__":
    main()