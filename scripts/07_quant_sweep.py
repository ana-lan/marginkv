# scripts/07_quant_sweep.py
"""Phase 5.4: fake-quant sweep on the doc-first LoRA cached path (MPS bf16).

Each doc KV is built once (build batch 1); every format in quant.FORMATS quantizes it (doc tokens
only, shared prefix BF16), scores all of the doc's pairs at score batch 8 with score_on_caches, and
records per-layer reconstruction error. All formats see identical pairs in identical batches.

Baseline: results/scores/qwen3_lora_dfB_cached_p5_b8_{ds}.json (5.2). The in-run bf16 arm must equal
it exactly. Verdicts use the 5.2 pre-registered rule and margins (results/metrics/quant_noise_floor.json).
Headline INT4 = int4_kivi; int4_tok is a diagnostic arm (pre-registered in 5.3).
Storage here is theory_bytes (payload only); real packed bytes come from 5.5.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse

import numpy as np
import torch

from marginkv.config import load_config
from marginkv.data.beir_loader import load_beir
from marginkv.eval import paired
from marginkv.eval.rerank import evaluate_rerank
from marginkv.kv import cache_ops as co
from marginkv.kv import quant as Q
from marginkv.kv import sweep
from marginkv.models import qwen3_reranker as qr
from marginkv.train.data import content_sha256, sha256_file
from marginkv.utils.device import get_device, get_dtype
from marginkv.utils.io import ensure_dir, load_json, load_pt, run_info, save_json, save_pt
from marginkv.utils.seed import set_seed
from marginkv.utils.timing import Timer

FORMATS = Q.FORMATS
SCORE_BATCH = 8
RULES = {
    "baseline": "BF16 cached, score batch 8 (5.2 p5_b8 file); in-run bf16 arm must match it exactly",
    "verdict": "no_measurable_loss if 95% CI of delta nDCG@10 inside [-m_ds, +m_ds] (5.2 margins); "
               "within_1_point if point delta >= -0.01; else loss",
    "headline_int4": "int4_kivi (int4_tok is diagnostic)",
    "storage": "theory_bytes payload here; packed bytes in 5.5",
}


def verify_model(model_dir):
    shas = load_json(model_dir / "marginkv_merge.json")["files_sha256"]
    for fname, want in shas.items():
        if sha256_file(model_dir / fname) != want:
            raise SystemExit(f"sha mismatch: {fname}")
    return shas


def make_transform(fmt, state, device):
    if fmt == "bf16":
        return sweep.identity

    def f(kv):
        with Timer(device) as t:
            q = Q.quantize_doc(kv, fmt)
        state["extra"].setdefault("quant_ms", {}).setdefault(fmt, []).append(t.ms)
        return q
    return f


def on_doc(state, name, d, base, kv):
    if name == "bf16":
        return
    st = Q.recon_stats(base, kv)
    L = len(st["k_err2"])
    R = state["extra"]["recon"].setdefault(name, {
        "k_err2": [0.0] * L, "k_ref2": [0.0] * L, "v_err2": [0.0] * L, "v_ref2": [0.0] * L,
        "k_maxabs": [0.0] * L, "v_maxabs": [0.0] * L, "doc_k_rel": [], "doc_v_rel": []})
    for key in ("k_err2", "k_ref2", "v_err2", "v_ref2"):
        R[key] = [a + b for a, b in zip(R[key], st[key])]
    for key in ("k_maxabs", "v_maxabs"):
        R[key] = [max(a, b) for a, b in zip(R[key], st[key])]
    R["doc_k_rel"].append((sum(st["k_err2"]) / sum(st["k_ref2"])) ** 0.5)
    R["doc_v_rel"].append((sum(st["v_err2"]) / sum(st["v_ref2"])) ** 0.5)


def recon_summary(R):
    L = len(R["k_err2"])
    k = [(R["k_err2"][l] / R["k_ref2"][l]) ** 0.5 for l in range(L)]
    v = [(R["v_err2"][l] / R["v_ref2"][l]) ** 0.5 for l in range(L)]
    return {"k_rel": (sum(R["k_err2"]) / sum(R["k_ref2"])) ** 0.5,
            "v_rel": (sum(R["v_err2"]) / sum(R["v_ref2"])) ** 0.5,
            "k_rel_per_layer": k, "v_rel_per_layer": v,
            "k_worst_layer": int(np.argmax(k)), "v_worst_layer": int(np.argmax(v)),
            "k_maxabs": max(R["k_maxabs"]), "v_maxabs": max(R["v_maxabs"]),
            "doc_k_rel_p50": float(np.percentile(R["doc_k_rel"], 50)),
            "doc_k_rel_p95": float(np.percentile(R["doc_k_rel"], 95)),
            "doc_v_rel_p50": float(np.percentile(R["doc_v_rel"], 50)),
            "doc_v_rel_p95": float(np.percentile(R["doc_v_rel"], 95)),
            "num_docs": len(R["doc_k_rel"])}


def score_record(meta, state, fmt, qids, art):
    out = state["out"][fmt]
    nest = lambda k: {q: {d: round(out[q][d][k], 6) for d in art["rerank_candidates"][q]} for q in qids}
    n_pairs = sum(len(out[q]) for q in qids)
    qms = state["extra"].get("quant_ms", {}).get(fmt, [])
    rec = {**meta, "format": fmt, "settings": {**meta["settings"], "score_batch": SCORE_BATCH},
           "num_pairs": n_pairs, "score_key": "log_odds", "aux_key": "p_yes",
           "scores": nest(0), "logit_yes": nest(1), "logit_no": nest(2), "aux": nest(3),
           "timing": {"build_total_s": sum(state["build_ms"]) / 1000,
                      "build_ms_per_doc": float(np.mean(state["build_ms"])),
                      "quant_ms_per_doc": float(np.mean(qms)) if qms else 0.0,
                      "score_total_s": sum(state["score_ms"][fmt]) / 1000,
                      "score_ms_per_pair": sum(state["score_ms"][fmt]) / n_pairs}}
    if fmt != "bf16":
        rec["recon"] = recon_summary(state["extra"]["recon"][fmt])
    return rec


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
    metrics_path = (A / "smoke" / "metrics" if smoke else R / "metrics") / "quant.json"
    depth = cfg.rerank.depth

    nf_path = R / "metrics" / "quant_noise_floor.json"
    if nf_path.exists():
        margins = load_json(nf_path)["margins"]
    elif smoke:
        margins = {ds: 0.002 for ds in cfg.datasets}
    else:
        raise SystemExit("run scripts/07a_noise_floor.py first (margins are pre-registered there)")

    model_dir = A / "lora" / "df" / "merged"
    shas = verify_model(model_dir)
    model = None

    report = {"run_info": run_info(), "phase": "5.4", "rules": RULES, "formats": list(FORMATS),
              "margins": margins, "smoke_queries": args.smoke, "datasets": {}}

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
                 "score_batch": SCORE_BATCH, "formats": list(FORMATS), "device": str(device),
                 "dtype": cfg.runtime.dtype}
        paths = {f: score_dir / f"qwen3_lora_dfB_cached_q_{f}_{ds}.json" for f in FORMATS}
        partial = part_dir / f"quant_{ds}.pt"

        if all(p.exists() for p in paths.values()) and not args.force:
            for p in paths.values():
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
            if state is None:
                state = sweep.new_state({f: None for f in FORMATS})
            state["extra"].setdefault("recon", {})
            configs = {f: (make_transform(f, state, device), SCORE_BATCH) for f in FORMATS}

            def on_chunk(st, done, total):
                save_pt({"guard": guard, "state": st}, partial)
                if done % 20 == 0 or done == total:
                    sm = " ".join(f"{f}:{np.mean(v):.0f}" for f, v in st["score_ms"].items())
                    qm = " ".join(f"{f}:{np.mean(v):.0f}" for f, v in st["extra"].get("quant_ms", {}).items())
                    print(f"[{ds}] chunk {done}/{total}  build {np.mean(st['build_ms']):.0f} ms/doc | "
                          f"score ms/batch {sm} | quant ms/doc {qm}", flush=True)

            state = sweep.run_sweep(model, art, dfart, device, configs, qids=qids,
                                    doc_chunk=args.doc_chunk, state=state, on_chunk=on_chunk, on_doc=on_doc)
            meta = {"run_info": run_info(), "phase": "5.4", "model_dir": str(model_dir), "model_files_sha256": shas,
                    "dataset": ds, "split": cfg.split, "variant": "B", "prompt_order": "doc_first", "path": "cached",
                    "device": str(device), "dtype": cfg.runtime.dtype, "depth": depth,
                    "settings": {"doc_chunk": args.doc_chunk, "build_batch": 1},
                    "inputs_sha256": inputs_sha, "smoke_queries": args.smoke,
                    "num_docs": len(state["build_ms"]), "query_order": qids}
            for f in FORMATS:
                save_json(score_record(meta, state, f, qids, art), paths[f])
            partial.unlink(missing_ok=True)

        recs = {f: load_json(paths[f]) for f in FORMATS}
        scores = {f: recs[f]["scores"] for f in FORMATS}
        base_path = R / "scores" / f"qwen3_lora_dfB_cached_p5_b8_{ds}.json"
        if smoke:
            base = scores["bf16"]
            bf16_check = None
        else:
            base = load_json(base_path)["scores"]
            diffs = [abs(scores["bf16"][q][d] - base[q][d]) for q in qids for d in base[q]]
            bf16_check = {"max_abs_vs_p5_b8": max(diffs), "exact": max(diffs) == 0.0}
            assert bf16_check["exact"], f"in-run bf16 differs from the 5.2 baseline (max {max(diffs):.2e})"

        ev_base = evaluate_rerank(cands, base, qrels, depth)
        _, docs, _ = sweep.sweep_plan(art, dfart, qids)
        shared_len = co.shared_prefix_len({**art["pieces"], **dfart["seam_pieces"]})
        n_docs = [dfart["cache_len_seam"][d] - shared_len for d in docs]
        bf_bytes = float(np.mean([Q.theory_bytes("bf16", n) for n in n_docs]))

        dsr = {"num_queries": len(qids), "num_docs": len(docs), "margin": margins[ds], "bf16_check": bf16_check,
               "baseline": {"mean": ev_base["mean"]}, "formats": {}}
        print(f"[{ds}] baseline nDCG {ev_base['mean']['ndcg@10']:.4f} MRR {ev_base['mean']['mrr@10']:.4f}  "
              f"margin {margins[ds]:.4f}  bf16 rerun exact: {bf16_check['exact'] if bf16_check else 'n/a (smoke)'}")
        for f in FORMATS:
            ev = evaluate_rerank(cands, scores[f], qrels, depth)
            cmp = paired.compare(ev_base["per_query"], ev["per_query"], qids, cfg.seed)
            agr = paired.score_agreement(cands, base, scores[f], qids, depth)
            theory = float(np.mean([Q.theory_bytes(f, n) for n in n_docs]))
            entry = {"mean": ev["mean"], "vs_bf16": cmp, "scores_vs_bf16": agr,
                     "verdict": paired.verdict(cmp["delta_ndcg@10"], cmp["ndcg@10_ci95"], margins[ds]),
                     "theory_bytes_per_doc": theory, "theory_ratio_vs_bf16": bf_bytes / theory,
                     "timing": recs[f]["timing"], "recon": recs[f].get("recon"),
                     "diagnostics": ev["diagnostics"]}
            dsr["formats"][f] = entry
            rc = entry["recon"]
            rec_s = f"K {rc['k_rel']:.4f} V {rc['v_rel']:.4f} (worst L{rc['k_worst_layer']}/L{rc['v_worst_layer']})" if rc else "-"
            ci = cmp["ndcg@10_ci95"]
            print(f"[{ds}] {f:10s} nDCG {ev['mean']['ndcg@10']:.4f} d {cmp['delta_ndcg@10']:+.4f} "
                  f"[{ci[0]:+.4f}, {ci[1]:+.4f}] dMRR {cmp['delta_mrr@10']:+.4f} W/L/T {cmp['wlt_ndcg@10']} | "
                  f"{entry['verdict']:18s} | {entry['theory_ratio_vs_bf16']:.2f}x | scores max {agr['max_abs']:.3f} "
                  f"flips {agr['pair_flip_rate']:.1e} | {rec_s}")
        report["datasets"][ds] = dsr

    save_json(report, metrics_path)
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()