# scripts/10_combined_sweep.py
"""Phase 7.2/7.3: combined compression (uniform 75% pruning, then FP8 / INT4 KIVI) on the doc-first
LoRA cached path (MPS bf16).

Same harness as 07_quant_sweep.py and 09_prune_sweep.py: each doc KV is built once (build batch 1),
every arm transforms it (kv.combined: prune doc tokens, then quantize kept doc tokens; shared prefix
BF16; next_pos unchanged) and scores all of the doc's pairs at score batch 8 with score_on_caches.

2x2 design per format: baseline (keep_all), prune only (uniform_r75), quant only (fp8_tensor /
int4_kivi), combined. All four are scored in this run. Check arms must reproduce earlier files
exactly. RULES below were committed before the full run. Packed sizes and the frontier come in 7.4.
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
from marginkv.kv import combined as CB
from marginkv.kv import prune as P
from marginkv.kv import quant as Q
from marginkv.kv import sweep
from marginkv.models import qwen3_reranker as qr
from marginkv.train.data import content_sha256, sha256_file
from marginkv.utils.device import get_device, get_dtype
from marginkv.utils.io import ensure_dir, load_json, load_pt, run_info, save_json, save_pt
from marginkv.utils.seed import set_seed

PHASE = "7.3"
SCORE_BATCH = 8
PRUNE_ARM = "uniform_r75"
FORMATS = ("fp8_tensor", "int4_kivi")

ARMS = {
    "keep_all": ("uniform", 1.0, "bf16"),
    "uniform_r75": ("uniform", 0.75, "bf16"),
    "fp8_tensor": ("uniform", 1.0, "fp8_tensor"),
    "int4_kivi": ("uniform", 1.0, "int4_kivi"),
    "uniform_r75_fp8_tensor": ("uniform", 0.75, "fp8_tensor"),
    "uniform_r75_int4_kivi": ("uniform", 0.75, "int4_kivi"),
}
COMBINED = {"uniform_r75_fp8_tensor": ("uniform_r75", "fp8_tensor"),
            "uniform_r75_int4_kivi": ("uniform_r75", "int4_kivi")}
FROZEN = {"uniform_r75": "qwen3_lora_dfB_cached_p6_uniform_r75_{ds}.json",
          "fp8_tensor": "qwen3_lora_dfB_cached_q_fp8_tensor_{ds}.json",
          "int4_kivi": "qwen3_lora_dfB_cached_q_int4_kivi_{ds}.json"}
P6_FILE = "qwen3_lora_dfB_cached_p6_{arm}_{ds}.json"
NAMED = {  # name: (a, b), paired delta = b - a; "p6:<arm>" = frozen Phase 6 score file
    "r75_int4_vs_int4": ("int4_kivi", "uniform_r75_int4_kivi"),
    "r75_fp8_vs_int4": ("int4_kivi", "uniform_r75_fp8_tensor"),
    "r75_int4_vs_uniform_r25": ("p6:uniform_r25", "uniform_r75_int4_kivi"),
}

RULES = {
    "phase": "7.2, pre-registered and committed before the full run (an 8-query smoke run was seen)",
    "setting": "uniform 75% pruning (Phase 6 pre-registered pick, asserted from prune.json) x {fp8_tensor, "
               "int4_kivi}; order prune then quantize (kv.combined); shared prefix BF16; next_pos unchanged; "
               "KIVI key groups over kept tokens",
    "baseline": "BF16 cached, score batch 8 (5.2 p5_b8 file). keep_all must match it exactly or the run aborts",
    "check_arms": "uniform_r75, fp8_tensor, int4_kivi are re-scored in-run and must equal their Phase 6 / Phase 5 "
                  "files exactly; a mismatch is flagged and reported (no abort); all comparisons use in-run arms",
    "verdict": "each arm vs p5_b8: no_measurable_loss if 95% CI of delta nDCG@10 inside [-m_ds, +m_ds] (5.2 "
               "margins); within_1_point if point delta >= -0.01; else loss",
    "interaction": "per query I = (comb - prune) - (quant - base) on nDCG@10; CI via paired.interaction "
                   "(percentile bootstrap, 10k paired query resamples, seed = cfg.seed). direction: compounds iff "
                   "CI upper < 0 on both datasets, sub_additive iff CI lower > 0 on both, else not_shown. size per "
                   "dataset: no_measurable if CI inside [-2 m_ds, +2 m_ds], within_1_point if CI inside "
                   "[-0.01, +0.01], else wide",
    "vs_single": "paired delta nDCG@10 of combined vs prune-only and vs quant-only (b - a); b_better iff CI lower > 0 "
                 "on both datasets, b_worse iff CI upper < 0 on both, else no_clear_difference",
    "named": "r75_int4_vs_int4 (int4_kivi -> uniform_r75_int4_kivi), r75_fp8_vs_int4 (int4_kivi -> "
             "uniform_r75_fp8_tensor), r75_int4_vs_uniform_r25 (frozen Phase 6 uniform_r25 -> "
             "uniform_r75_int4_kivi); labels as vs_single",
    "frontier_7_4": "with packed MiB/doc per dataset (7.4): single-method pool = Phase 5 formats incl. bf16 + Phase 6 "
                    "non-diagnostic arms. combined arm C is dominated iff some single X with MiB(X) <= MiB(C) on both "
                    "datasets has paired CI of (X - C) lower > 0 on both; on_frontier iff for every single X with "
                    "MiB(X) <= MiB(C) on both, CI of (C - X) lower > 0 on both; else not_separated. The point-estimate "
                    "frontier is reported as descriptive only",
    "bootstrap": "paired.compare / paired.interaction: percentile bootstrap, 10k paired query resamples, seed = cfg.seed",
    "storage": "theory bytes (kv.quant.theory_bytes) of kept doc tokens here; packed bytes in 7.4",
}


def verify_model(model_dir):
    shas = load_json(model_dir / "marginkv_merge.json")["files_sha256"]
    for fname, want in shas.items():
        if sha256_file(model_dir / fname) != want:
            raise SystemExit(f"sha mismatch: {fname}")
    return shas


def on_doc(state, name, d, base, kv):
    if kv.next_pos != base.next_pos or kv.shared_len != base.shared_len:
        raise SystemExit(f"{name} {d}: next_pos/shared_len changed")
    s = kv.shared_len
    if not (torch.equal(kv.keys[0][:, :s], base.keys[0][:, :s]) and torch.equal(kv.values[0][:, :s], base.values[0][:, :s])):
        raise SystemExit(f"{name} {d}: shared prefix modified")
    state["extra"].setdefault("retention", {}).setdefault(name, {})[d] = [P.doc_len(kv), kv.n_cached - s]


def retention_summary(R, retention):
    Ts = np.array([v[0] for v in R.values()])
    kept = np.array([v[1] for v in R.values()])
    mand = np.array([len(P.mandatory_idx(int(t))) for t in Ts])
    targets = np.array([P.round_half_up(retention * t) for t in Ts])
    expect = np.minimum(Ts, np.maximum(targets, mand))
    assert np.array_equal(kept, expect), "kept tokens differ from the budget rule"
    return {"target_retention": retention,
            "actual_retention_mean": float((kept / Ts).mean()),
            "actual_retention_tokens": float(kept.sum() / Ts.sum()),
            "forced_frac": float((kept > targets).mean()),
            "kept_tokens_mean": float(kept.mean()), "doc_tokens_mean": float(Ts.mean()),
            "num_docs": int(len(Ts))}


def score_record(meta, state, arm, qids, art):
    out = state["out"][arm]
    nest = lambda k: {q: {d: round(out[q][d][k], 6) for d in art["rerank_candidates"][q]} for q in qids}
    n_pairs = sum(len(out[q]) for q in qids)
    m, r, f = ARMS[arm]
    return {**meta, "arm": arm, "method": m, "retention": r, "format": f, "order": CB.ORDER,
            "settings": {**meta["settings"], "score_batch": SCORE_BATCH},
            "num_pairs": n_pairs, "score_key": "log_odds", "aux_key": "p_yes",
            "scores": nest(0), "logit_yes": nest(1), "logit_no": nest(2), "aux": nest(3),
            "retention_per_doc": state["extra"]["retention"][arm],
            "timing": {"build_total_s": sum(state["build_ms"]) / 1000,
                       "build_ms_per_doc": float(np.mean(state["build_ms"])),
                       "score_total_s": sum(state["score_ms"][arm]) / 1000,
                       "score_ms_per_pair": sum(state["score_ms"][arm]) / n_pairs}}


def exact_check(scores, ref_path, qids):
    if not ref_path.exists():
        raise SystemExit(f"missing {ref_path}; Phase 6 files: tar xzf results/scores/p6_scores.tar.gz (repo root)")
    ref = load_json(ref_path)["scores"]
    diffs = [abs(scores[q][d] - ref[q][d]) for q in qids for d in ref[q]]
    return {"file": ref_path.name, "max_abs": max(diffs), "exact": max(diffs) == 0.0}


def diff_label(cis):
    if all(c[0] > 0 for c in cis):
        return "b_better"
    if all(c[1] < 0 for c in cis):
        return "b_worse"
    return "no_clear_difference"


def interaction_label(dsets, arm):
    per = {ds: dsets[ds]["interaction"][arm] for ds in dsets}
    cis = [per[ds]["ndcg@10_ci95"] for ds in dsets]
    if all(c[1] < 0 for c in cis):
        direction = "compounds"
    elif all(c[0] > 0 for c in cis):
        direction = "sub_additive"
    else:
        direction = "not_shown"
    size = {}
    for ds in dsets:
        lo, hi = per[ds]["ndcg@10_ci95"]
        m2 = 2 * dsets[ds]["margin"]
        if lo >= -m2 and hi <= m2:
            size[ds] = "no_measurable"
        elif lo >= -paired.ONE_POINT and hi <= paired.ONE_POINT:
            size[ds] = "within_1_point"
        else:
            size[ds] = "wide"
    return {"direction": direction, "size": size,
            "per_ds": {ds: [per[ds]["interaction_ndcg@10"], per[ds]["ndcg@10_ci95"]] for ds in dsets}}


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
    metrics_path = (A / "smoke" / "metrics" if smoke else R / "metrics") / "combined.json"
    ensure_dir(metrics_path.parent)
    depth = cfg.rerank.depth

    margins = load_json(R / "metrics" / "quant_noise_floor.json")["margins"]
    sel = load_json(R / "metrics" / "prune.json")["summary"]["phase7_selection"]
    assert sel["arm"] == PRUNE_ARM, f"prune.json selects {sel['arm']}, expected {PRUNE_ARM}"

    model_dir = A / "lora" / "df" / "merged"
    shas = verify_model(model_dir)
    model = None
    rules_sha = content_sha256(RULES)

    report = {"run_info": run_info(), "phase": PHASE, "rules": RULES, "rules_sha256": rules_sha,
              "arms": {a: list(v) for a, v in ARMS.items()}, "order": CB.ORDER, "margins": margins,
              "phase6_selection": sel, "smoke_queries": args.smoke, "datasets": {}}

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
                 "score_batch": SCORE_BATCH, "arms": {a: list(v) for a, v in ARMS.items()}, "rules_sha256": rules_sha,
                 "device": str(device), "dtype": cfg.runtime.dtype}
        paths = {a: score_dir / f"qwen3_lora_dfB_cached_p7_{a}_{ds}.json" for a in ARMS}
        partial = part_dir / f"combined_{ds}.pt"

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
                state = sweep.new_state({a: None for a in ARMS})
            configs = {a: (CB.make_combined(m, r, f), SCORE_BATCH) for a, (m, r, f) in ARMS.items()}

            def on_chunk(st, done, total):
                save_pt({"guard": guard, "state": st}, partial)
                if done % 20 == 0 or done == total:
                    sm = " ".join(f"{a}:{np.mean(v):.0f}" for a, v in st["score_ms"].items() if v)
                    print(f"[{ds}] chunk {done}/{total}  build {np.mean(st['build_ms']):.0f} ms/doc | "
                          f"score ms/batch {sm}", flush=True)

            state = sweep.run_sweep(model, art, dfart, device, configs, qids=qids,
                                    doc_chunk=args.doc_chunk, state=state, on_chunk=on_chunk, on_doc=on_doc)
            meta = {"run_info": run_info(), "phase": PHASE, "model_dir": str(model_dir), "model_files_sha256": shas,
                    "dataset": ds, "split": cfg.split, "variant": "B", "prompt_order": "doc_first", "path": "cached",
                    "device": str(device), "dtype": cfg.runtime.dtype, "depth": depth,
                    "settings": {"doc_chunk": args.doc_chunk, "build_batch": 1},
                    "inputs_sha256": inputs_sha, "smoke_queries": args.smoke,
                    "rules_sha256": rules_sha, "num_docs": len(state["build_ms"]), "query_order": qids}
            for a in ARMS:
                save_json(score_record(meta, state, a, qids, art), paths[a])
            partial.unlink(missing_ok=True)

        recs = {a: load_json(paths[a]) for a in ARMS}
        scores = {a: recs[a]["scores"] for a in ARMS}
        for a in COMBINED:
            assert recs[a]["retention_per_doc"] == recs[PRUNE_ARM]["retention_per_doc"], f"{a}: kept set sizes differ"

        checks = {}
        if smoke:
            base = scores["keep_all"]
        else:
            base = load_json(R / "scores" / f"qwen3_lora_dfB_cached_p5_b8_{ds}.json")["scores"]
            checks["keep_all"] = exact_check(scores["keep_all"], R / "scores" / f"qwen3_lora_dfB_cached_p5_b8_{ds}.json", qids)
            assert checks["keep_all"]["exact"], f"keep_all differs from the 5.2 baseline (max {checks['keep_all']['max_abs']:.2e})"
            for a, fn in FROZEN.items():
                checks[a] = exact_check(scores[a], R / "scores" / fn.format(ds=ds), qids)
                if not checks[a]["exact"]:
                    print(f"[{ds}] FLAG: {a} differs from {checks[a]['file']} (max {checks[a]['max_abs']:.2e})")

        ev_base = evaluate_rerank(cands, base, qrels, depth)
        evs = {a: evaluate_rerank(cands, scores[a], qrels, depth) for a in ARMS}
        ext = {}
        for name, (a, b) in NAMED.items():
            for x in (a, b):
                if x.startswith("p6:") and x not in ext:
                    s6 = load_json(R / "scores" / P6_FILE.format(arm=x[3:], ds=ds))["scores"]
                    ext[x] = evaluate_rerank(cands, {q: s6[q] for q in qids}, qrels, depth)
        pq = lambda x: (ext[x] if x.startswith("p6:") else evs[x])["per_query"]

        dsr = {"num_queries": len(qids), "num_docs": recs["keep_all"]["num_docs"], "margin": margins[ds],
               "checks": checks or None, "baseline": {"mean": ev_base["mean"]}, "arms": {},
               "interaction": {}, "vs_prune_only": {}, "vs_quant_only": {}, "named": {}}
        flags = " ".join(f"{a}:{'exact' if c['exact'] else 'FLAG'}" for a, c in checks.items()) or "n/a (smoke)"
        print(f"[{ds}] baseline nDCG {ev_base['mean']['ndcg@10']:.4f} MRR {ev_base['mean']['mrr@10']:.4f}  "
              f"margin {margins[ds]:.4f}  checks: {flags}")

        for a, (m, r, f) in ARMS.items():
            if a == "keep_all":
                continue
            ev = evs[a]
            cmp = paired.compare(ev_base["per_query"], ev["per_query"], qids, cfg.seed)
            agr = paired.score_agreement(cands, base, scores[a], qids, depth)
            ret = retention_summary(recs[a]["retention_per_doc"], r)
            rpd = list(recs[a]["retention_per_doc"].values())
            kept_b = float(np.mean([Q.theory_bytes(f, k) for _, k in rpd]))
            full_b = float(np.mean([Q.theory_bytes("bf16", t) for t, _ in rpd]))
            entry = {"method": m, "retention": r, "format": f, "combined": a in COMBINED, "mean": ev["mean"],
                     "vs_baseline": cmp, "scores_vs_baseline": agr,
                     "verdict": paired.verdict(cmp["delta_ndcg@10"], cmp["ndcg@10_ci95"], margins[ds]),
                     "retention_stats": ret, "theory_bytes_per_doc": kept_b, "theory_ratio_vs_bf16": full_b / kept_b,
                     "timing": recs[a]["timing"], "diagnostics": ev["diagnostics"]}
            dsr["arms"][a] = entry
            ci = cmp["ndcg@10_ci95"]
            print(f"[{ds}] {a:23s} nDCG {ev['mean']['ndcg@10']:.4f} d {cmp['delta_ndcg@10']:+.4f} "
                  f"[{ci[0]:+.4f}, {ci[1]:+.4f}] dMRR {cmp['delta_mrr@10']:+.4f} W/L/T {cmp['wlt_ndcg@10']} | "
                  f"{entry['verdict']:18s} | ret {ret['actual_retention_tokens']:.3f} | "
                  f"{entry['theory_ratio_vs_bf16']:.2f}x | scores max {agr['max_abs']:.3f} flips {agr['pair_flip_rate']:.1e}")

        for a, (pa, qa) in COMBINED.items():
            it = paired.interaction(ev_base["per_query"], evs[pa]["per_query"], evs[qa]["per_query"],
                                    evs[a]["per_query"], qids, cfg.seed)
            dsr["interaction"][a] = it
            dsr["vs_prune_only"][a] = paired.compare(evs[pa]["per_query"], evs[a]["per_query"], qids, cfg.seed)
            dsr["vs_quant_only"][a] = paired.compare(evs[qa]["per_query"], evs[a]["per_query"], qids, cfg.seed)
            ci = it["ndcg@10_ci95"]
            print(f"[{ds}] interaction {a:23s} I {it['interaction_ndcg@10']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
                  f"(prune {it['delta_prune_ndcg@10']:+.4f} + quant {it['delta_quant_ndcg@10']:+.4f} = "
                  f"{it['additive_prediction_ndcg@10']:+.4f}; observed {it['delta_comb_ndcg@10']:+.4f})")
        for name, (a, b) in NAMED.items():
            c = paired.compare(pq(a), pq(b), qids, cfg.seed)
            dsr["named"][name] = {"a": a, "b": b, **c}
            ci = c["ndcg@10_ci95"]
            print(f"[{ds}] named {name:24s} d {c['delta_ndcg@10']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        report["datasets"][ds] = dsr

    dsets = report["datasets"]
    if not smoke and set(dsets) == set(cfg.datasets):
        cis = lambda key, a: [dsets[ds][key][a]["ndcg@10_ci95"] for ds in dsets]
        summ = {"combined": {}, "named": {}, "checks_all_exact": all(c["exact"] for ds in dsets for c in dsets[ds]["checks"].values())}
        for a in COMBINED:
            summ["combined"][a] = {
                "verdict": {ds: dsets[ds]["arms"][a]["verdict"] for ds in dsets},
                "interaction": interaction_label(dsets, a),
                "vs_prune_only": diff_label(cis("vs_prune_only", a)),
                "vs_quant_only": diff_label(cis("vs_quant_only", a))}
            s = summ["combined"][a]
            print(f"[summary] {a:23s} verdict {s['verdict']} | interaction {s['interaction']['direction']} "
                  f"{s['interaction']['size']} | vs prune-only {s['vs_prune_only']} | vs quant-only {s['vs_quant_only']}")
        for name in NAMED:
            summ["named"][name] = {"label": diff_label(cis("named", name)),
                                   "per_ds": {ds: [dsets[ds]["named"][name]["delta_ndcg@10"], dsets[ds]["named"][name]["ndcg@10_ci95"]] for ds in dsets}}
            print(f"[summary] named {name:24s} {summ['named'][name]['label']}")
        print(f"[summary] all check arms exact: {summ['checks_all_exact']}")
        report["summary"] = summ

    save_json(report, metrics_path)
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()