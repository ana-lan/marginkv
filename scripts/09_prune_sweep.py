# scripts/09_prune_sweep.py
"""Phase 6.3/6.4: query-independent KV pruning sweep on the doc-first LoRA cached path (MPS bf16).

Same harness as 07_quant_sweep.py: each doc KV is built once (build batch 1); every arm prunes it
(doc tokens only, shared prefix untouched, next_pos unchanged) and scores all of the doc's pairs at
score batch 8 with score_on_caches. All arms see identical pairs in identical batch order.

Baseline: results/scores/qwen3_lora_dfB_cached_p5_b8_{ds}.json (5.2). keep_all (uniform at retention
1.0 through the pruning path) must equal it exactly. Importance from 6.2 (artifacts/prune/importance_{ds}.pt).
Verdicts use the 5.2 pre-registered rule and margins. RULES below were committed before the full run.
Storage here is theory BF16 bytes of the kept tokens; packed bytes come in 6.5.
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
from marginkv.kv import prune as P
from marginkv.kv import quant as Q
from marginkv.kv import sweep
from marginkv.models import qwen3_reranker as qr
from marginkv.train.data import content_sha256, sha256_file
from marginkv.utils.device import get_device, get_dtype
from marginkv.utils.io import ensure_dir, load_json, load_pt, run_info, save_json, save_pt
from marginkv.utils.seed import set_seed

PHASE = "6.4"
SCORE_BATCH = 8
MAIN_METHODS = ("uniform", "boundary", "attn_a", "attn_b", "random")
DIAG_METHODS = ("attn_h2o",)
SELECT_METHODS = ("uniform", "boundary", "attn_a", "attn_b")
ATTN_METHODS = ("attn_a", "attn_b", "attn_h2o")
RETENTIONS = P.RETENTIONS


def arm_name(m, r):
    return f"{m}_r{round(r * 100)}"


ARMS = {"keep_all": ("uniform", 1.0)}
for _m in MAIN_METHODS + DIAG_METHODS:
    for _r in RETENTIONS:
        ARMS[arm_name(_m, _r)] = (_m, _r)

RULES = {
    "phase": "6.3, pre-registered and committed before the full run",
    "baseline": "BF16 cached, score batch 8 (5.2 p5_b8 file). keep_all (uniform at retention 1.0 through the "
                "pruning path) must match it exactly (max diff 0) or the run aborts",
    "arms": "{uniform, boundary, attn_a, attn_b, random} x {0.25, 0.5, 0.75}; attn_h2o x 3 is diagnostic "
            "(reported, never selected)",
    "constants": {"first_keep": P.FIRST_KEEP, "reserve": P.RESERVE, "boundary_first": P.BOUNDARY_FIRST,
                  "random_seed": "doc_seed(cfg.seed, doc_id), one seed"},
    "importance": "6.2 artifacts: eager attention, mean over 16 query heads and 28 layers, doc rows only, raw "
                  "probabilities; (a) sum_{j>i} A(j,i)/(T-i-1); (b) same weighted by absolute position + 1; "
                  "h2o = (a) * (T-i-1) (plain column sum); ties -> lower index",
    "retention": "doc tokens only (shared prefix never pruned, not counted); kept = min(T, max(half-up(r*T), "
                 "mandatory)); mandatory = first 1 + last 8 doc tokens; report target, actual, forced fraction, "
                 "mandatory fraction",
    "verdict": "vs p5_b8: no_measurable_loss if 95% CI of delta nDCG@10 inside [-m_ds, +m_ds] (5.2 margins); "
               "within_1_point if point delta >= -0.01; else loss",
    "attn_vs_random": "paired delta nDCG@10 (attn - random) at equal retention; beats_random iff 95% CI lower "
                      "bound > 0 on both datasets",
    "b_vs_a": "paired delta nDCG@10 (attn_b - attn_a) at equal retention; helps iff CI lower > 0 on both datasets, "
              "hurts iff CI upper < 0 on both, else no_clear_difference",
    "phase7_selection": "methods uniform, boundary, attn_a, attn_b; lowest retention at which some method's verdict "
                        "is not loss on both datasets; ties -> larger mean delta nDCG@10 over datasets; if every "
                        "0.75 arm is a loss, pick the best 0.75 arm by mean delta and flag all_loss",
    "bootstrap": "paired.compare: percentile bootstrap, 10k paired query resamples, seed = cfg.seed",
    "storage": "theory BF16 bytes of kept doc tokens here; packed bytes in 6.5",
}


def verify_model(model_dir):
    shas = load_json(model_dir / "marginkv_merge.json")["files_sha256"]
    for fname, want in shas.items():
        if sha256_file(model_dir / fname) != want:
            raise SystemExit(f"sha mismatch: {fname}")
    return shas


def make_transforms(imp, seed):
    impd = {"attn_a": imp["imp_a"], "attn_b": imp["imp_b"],
            "attn_h2o": {d: P.importance_h2o(v) for d, v in imp["imp_a"].items()}}
    out = {}
    for name, (m, r) in ARMS.items():
        out[name] = P.make_transform(m, r, importance=impd.get(m), seed=seed if m == "random" else None)
    return out


def on_doc(state, name, d, base, kv):
    if kv.next_pos != base.next_pos or kv.shared_len != base.shared_len:
        raise SystemExit(f"{name} {d}: next_pos/shared_len changed")
    s = kv.shared_len
    if not torch.equal(kv.keys[0][:, :s], base.keys[0][:, :s]):
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
            "mandatory_frac_mean": float((mand / Ts).mean()),
            "kept_tokens_mean": float(kept.mean()), "doc_tokens_mean": float(Ts.mean()),
            "num_docs": int(len(Ts))}


def score_record(meta, state, arm, qids, art):
    out = state["out"][arm]
    nest = lambda k: {q: {d: round(out[q][d][k], 6) for d in art["rerank_candidates"][q]} for q in qids}
    n_pairs = sum(len(out[q]) for q in qids)
    m, r = ARMS[arm]
    return {**meta, "arm": arm, "method": m, "retention": r,
            "settings": {**meta["settings"], "score_batch": SCORE_BATCH},
            "num_pairs": n_pairs, "score_key": "log_odds", "aux_key": "p_yes",
            "scores": nest(0), "logit_yes": nest(1), "logit_no": nest(2), "aux": nest(3),
            "retention_per_doc": state["extra"]["retention"][arm],
            "timing": {"build_total_s": sum(state["build_ms"]) / 1000,
                       "build_ms_per_doc": float(np.mean(state["build_ms"])),
                       "score_total_s": sum(state["score_ms"][arm]) / 1000,
                       "score_ms_per_pair": sum(state["score_ms"][arm]) / n_pairs}}


def ci_label(ci_all, lo_rule, hi_rule):
    if all(c[0] > 0 for c in ci_all):
        return lo_rule
    if hi_rule and all(c[1] < 0 for c in ci_all):
        return hi_rule
    return "no_clear_difference" if hi_rule else "not_shown"


def select_phase7(dsets):
    dss = list(dsets)
    md = lambda a: float(np.mean([dsets[ds]["arms"][a]["vs_baseline"]["delta_ndcg@10"] for ds in dss]))
    for r in RETENTIONS:
        ok = [a for a in (arm_name(m, r) for m in SELECT_METHODS)
              if all(dsets[ds]["arms"][a]["verdict"] != "loss" for ds in dss)]
        if ok:
            ranked = sorted(ok, key=lambda a: (-md(a), a))
            return {"arm": ranked[0], "method": ARMS[ranked[0]][0], "retention": r,
                    "mean_delta_ndcg@10": md(ranked[0]), "candidates": ranked, "all_loss": False}
    r = RETENTIONS[-1]
    ranked = sorted((arm_name(m, r) for m in SELECT_METHODS), key=lambda a: (-md(a), a))
    return {"arm": ranked[0], "method": ARMS[ranked[0]][0], "retention": r,
            "mean_delta_ndcg@10": md(ranked[0]), "candidates": ranked, "all_loss": True}


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
    metrics_path = (A / "smoke" / "metrics" if smoke else R / "metrics") / "prune.json"
    ensure_dir(metrics_path.parent)
    depth = cfg.rerank.depth

    nf_path = R / "metrics" / "quant_noise_floor.json"
    margins = load_json(nf_path)["margins"]

    model_dir = A / "lora" / "df" / "merged"
    shas = verify_model(model_dir)
    model = None
    rules_sha = content_sha256(RULES)

    report = {"run_info": run_info(), "phase": PHASE, "rules": RULES, "rules_sha256": rules_sha,
              "arms": {a: list(v) for a, v in ARMS.items()}, "margins": margins,
              "smoke_queries": args.smoke, "datasets": {}}

    for ds in cfg.datasets:
        art = load_pt(A / "tokenized" / f"{ds}_qwen3.pt")
        dfart = load_pt(A / "tokenized" / f"{ds}_qwen3_df.pt")
        assert dfart["base_content_sha256"] == art["content_sha256"], "seam artifact mismatch"
        imp = load_pt(A / "prune" / f"importance_{ds}.pt")
        assert imp["base_content_sha256"] == art["content_sha256"] and imp["df_content_sha256"] == dfart["content_sha256"]
        assert imp["model_files_sha256"] == shas, "importance computed with a different model"
        qids = list(art["query_order"])[:args.smoke] if smoke else list(art["query_order"])
        cands_all = load_json(R / "candidates" / f"{ds}.json")["candidates"]
        cands = {q: cands_all[q] for q in qids}
        beir = load_beir(ds, cfg.split, data_dir=str(cfg.paths.data_dir), download=False)
        qrels = {q: beir.qrels[q] for q in qids}

        inputs_sha = content_sha256({"art": art["content_sha256"], "dfart": dfart["content_sha256"], "qids": qids})
        guard = {"inputs_sha256": inputs_sha, "model_files_sha256": shas, "doc_chunk": args.doc_chunk,
                 "score_batch": SCORE_BATCH, "arms": list(ARMS), "rules_sha256": rules_sha,
                 "importance_key": imp["key"], "device": str(device), "dtype": cfg.runtime.dtype}
        paths = {a: score_dir / f"qwen3_lora_dfB_cached_p6_{a}_{ds}.json" for a in ARMS}
        partial = part_dir / f"prune_{ds}.pt"

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
            tf = make_transforms(imp, cfg.seed)
            configs = {a: (tf[a], SCORE_BATCH) for a in ARMS}

            def on_chunk(st, done, total):
                save_pt({"guard": guard, "state": st}, partial)
                if done % 20 == 0 or done == total:
                    sm = " ".join(f"{a}:{np.mean(v):.0f}" for a, v in st["score_ms"].items() if a.endswith(("_r25", "keep_all")))
                    print(f"[{ds}] chunk {done}/{total}  build {np.mean(st['build_ms']):.0f} ms/doc | "
                          f"score ms/batch {sm}", flush=True)

            state = sweep.run_sweep(model, art, dfart, device, configs, qids=qids,
                                    doc_chunk=args.doc_chunk, state=state, on_chunk=on_chunk, on_doc=on_doc)
            meta = {"run_info": run_info(), "phase": PHASE, "model_dir": str(model_dir), "model_files_sha256": shas,
                    "dataset": ds, "split": cfg.split, "variant": "B", "prompt_order": "doc_first", "path": "cached",
                    "device": str(device), "dtype": cfg.runtime.dtype, "depth": depth,
                    "settings": {"doc_chunk": args.doc_chunk, "build_batch": 1},
                    "inputs_sha256": inputs_sha, "smoke_queries": args.smoke, "importance_key": imp["key"],
                    "rules_sha256": rules_sha, "num_docs": len(state["build_ms"]), "query_order": qids}
            for a in ARMS:
                save_json(score_record(meta, state, a, qids, art), paths[a])
            partial.unlink(missing_ok=True)

        recs = {a: load_json(paths[a]) for a in ARMS}
        scores = {a: recs[a]["scores"] for a in ARMS}
        if smoke:
            base = scores["keep_all"]
            keep_check = None
        else:
            base = load_json(R / "scores" / f"qwen3_lora_dfB_cached_p5_b8_{ds}.json")["scores"]
            diffs = [abs(scores["keep_all"][q][d] - base[q][d]) for q in qids for d in base[q]]
            keep_check = {"max_abs_vs_p5_b8": max(diffs), "exact": max(diffs) == 0.0}
            assert keep_check["exact"], f"keep_all differs from the 5.2 baseline (max {max(diffs):.2e})"

        ev_base = evaluate_rerank(cands, base, qrels, depth)
        evs = {a: evaluate_rerank(cands, scores[a], qrels, depth) for a in ARMS}
        dsr = {"num_queries": len(qids), "num_docs": recs["keep_all"]["num_docs"], "margin": margins[ds],
               "keep_all_check": keep_check, "baseline": {"mean": ev_base["mean"]}, "arms": {},
               "attn_vs_random": {}, "b_vs_a": {}}
        print(f"[{ds}] baseline nDCG {ev_base['mean']['ndcg@10']:.4f} MRR {ev_base['mean']['mrr@10']:.4f}  "
              f"margin {margins[ds]:.4f}  keep_all exact: {keep_check['exact'] if keep_check else 'n/a (smoke)'}")

        for a, (m, r) in ARMS.items():
            if a == "keep_all":
                continue
            ev = evs[a]
            cmp = paired.compare(ev_base["per_query"], ev["per_query"], qids, cfg.seed)
            agr = paired.score_agreement(cands, base, scores[a], qids, depth)
            ret = retention_summary(recs[a]["retention_per_doc"], r)
            rpd = recs[a]["retention_per_doc"].values()
            kept_b = float(np.mean([Q.theory_bytes("bf16", k) for _, k in rpd]))
            full_b = float(np.mean([Q.theory_bytes("bf16", t) for t, _ in rpd]))
            entry = {"method": m, "retention": r, "diagnostic": m in DIAG_METHODS, "mean": ev["mean"],
                     "vs_baseline": cmp, "scores_vs_baseline": agr,
                     "verdict": paired.verdict(cmp["delta_ndcg@10"], cmp["ndcg@10_ci95"], margins[ds]),
                     "retention_stats": ret, "theory_bytes_per_doc": kept_b, "theory_ratio_vs_bf16": full_b / kept_b,
                     "timing": recs[a]["timing"], "diagnostics": ev["diagnostics"]}
            dsr["arms"][a] = entry
            ci = cmp["ndcg@10_ci95"]
            print(f"[{ds}] {a:15s} nDCG {ev['mean']['ndcg@10']:.4f} d {cmp['delta_ndcg@10']:+.4f} "
                  f"[{ci[0]:+.4f}, {ci[1]:+.4f}] dMRR {cmp['delta_mrr@10']:+.4f} W/L/T {cmp['wlt_ndcg@10']} | "
                  f"{entry['verdict']:18s} | ret {ret['actual_retention_tokens']:.3f} forced {ret['forced_frac']:.2f} | "
                  f"{entry['theory_ratio_vs_bf16']:.2f}x | scores max {agr['max_abs']:.3f} flips {agr['pair_flip_rate']:.1e}")

        for r in RETENTIONS:
            rnd = evs[arm_name("random", r)]["per_query"]
            for m in ATTN_METHODS:
                dsr["attn_vs_random"][arm_name(m, r)] = paired.compare(rnd, evs[arm_name(m, r)]["per_query"], qids, cfg.seed)
            dsr["b_vs_a"][str(r)] = paired.compare(evs[arm_name("attn_a", r)]["per_query"],
                                                   evs[arm_name("attn_b", r)]["per_query"], qids, cfg.seed)
        report["datasets"][ds] = dsr

    dsets = report["datasets"]
    if not smoke and set(dsets) == set(cfg.datasets):
        summ = {"attn_vs_random": {}, "b_vs_a": {}}
        for r in RETENTIONS:
            for m in ATTN_METHODS:
                a = arm_name(m, r)
                cis = [dsets[ds]["attn_vs_random"][a]["ndcg@10_ci95"] for ds in dsets]
                summ["attn_vs_random"][a] = {
                    "label": "beats_random" if all(c[0] > 0 for c in cis) else "not_shown",
                    "per_ds": {ds: [dsets[ds]["attn_vs_random"][a]["delta_ndcg@10"], dsets[ds]["attn_vs_random"][a]["ndcg@10_ci95"]] for ds in dsets}}
            cis = [dsets[ds]["b_vs_a"][str(r)]["ndcg@10_ci95"] for ds in dsets]
            summ["b_vs_a"][str(r)] = {
                "label": ci_label(cis, "helps", "hurts"),
                "per_ds": {ds: [dsets[ds]["b_vs_a"][str(r)]["delta_ndcg@10"], dsets[ds]["b_vs_a"][str(r)]["ndcg@10_ci95"]] for ds in dsets}}
        summ["phase7_selection"] = select_phase7(dsets)
        report["summary"] = summ
        for a, v in summ["attn_vs_random"].items():
            print(f"[summary] {a:15s} vs random: {v['label']}  " + "  ".join(f"{ds} {d:+.4f} [{c[0]:+.4f}, {c[1]:+.4f}]" for ds, (d, c) in v["per_ds"].items()))
        for r, v in summ["b_vs_a"].items():
            print(f"[summary] b vs a @{r}: {v['label']}  " + "  ".join(f"{ds} {d:+.4f} [{c[0]:+.4f}, {c[1]:+.4f}]" for ds, (d, c) in v["per_ds"].items()))
        s7 = summ["phase7_selection"]
        print(f"[summary] Phase 7 setting: {s7['arm']} (mean d {s7['mean_delta_ndcg@10']:+.4f}, all_loss {s7['all_loss']}); candidates {s7['candidates']}")

    save_json(report, metrics_path)
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()