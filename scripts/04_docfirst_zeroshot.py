# scripts/04_docfirst_zeroshot.py
"""Phase 3.2: zero-shot document-first scoring, full prompt (no KV cache).

Variants:
  A: prefix + df_head + doc + "\\n<Query>:" + query + suffix      (Phase 2 artifact)
  B: prefix + df_head + seam_doc + "<Query>:" + query + suffix    (Phase 3.1 artifact)

Same pairs, batch size, dtype and length-sorted batching as the query-first reference.
Chunked and resumable at batch granularity (batch composition is identical with or without resume).
--smoke N scores the first N queries and writes under artifacts/smoke/ only.

Outputs (full run):
  results/scores/qwen3_df{A,B}_{ds}.json
  results/metrics/docfirst_zeroshot.json  vs query-first: deltas, paired W/L/T, bootstrap CI,
                                          rank agreement, order gate; plus B vs A
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # must be set before transformers is imported

import argparse
import hashlib
import json
import time

import numpy as np

from marginkv.config import load_config
from marginkv.data.beir_loader import load_beir
from marginkv.eval.rerank import evaluate_rerank, identity_scores
from marginkv.models.qwen3_reranker import doc_first_ids, doc_first_seam_ids, load_model, score_batch
from marginkv.utils.device import get_device, get_dtype
from marginkv.utils.io import ensure_dir, load_json, load_pt, run_info, save_json
from marginkv.utils.seed import set_seed
from marginkv.utils.timing import Timer, summarize_ms

CHUNK_BATCHES = 16
KEYS = ("log_odds", "p_yes", "logit_yes", "logit_no")
GATE = 0.01   # order gate: doc-first nDCG@10 >= query-first - 0.01
BOOT = 1000
EPS = 1e-9


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _len_stats(xs) -> dict:
    a = np.asarray(xs)
    return {"min": int(a.min()), "mean": round(float(a.mean()), 2), "p50": int(np.percentile(a, 50)),
            "p95": int(np.percentile(a, 95)), "max": int(a.max())}


def build_inputs(art, dfart, variant, qids):
    top, q_df = art["rerank_candidates"], art["query_df"]
    pairs = [(q, d) for q in qids for d in top[q]]
    if variant == "A":
        seqs = [doc_first_ids(art["pieces"], art["doc"][d], q_df[q]) for q, d in pairs]
    else:
        pb = {**art["pieces"], **dfart["seam_pieces"]}
        seqs = [doc_first_seam_ids(pb, dfart["doc_seam"][d], q_df[q]) for q, d in pairs]
    return pairs, seqs


def score_run(model, seqs, batch_size, device, partial_path, meta, tag):
    """Length-sorted fixed batches (same rule as score_sequences). Saves state every CHUNK_BATCHES."""
    order = sorted(range(len(seqs)), key=lambda i: (len(seqs[i]), i))
    batches = [order[s:s + batch_size] for s in range(0, len(order), batch_size)]
    state = {"meta": meta, "batches": {}, "batch_ms": {}}
    if partial_path.exists():
        old = load_json(partial_path)
        if old.get("meta") != meta:
            raise SystemExit(f"{tag} partial {partial_path} has different inputs/settings. Delete it or use --force.")
        state = old
        print(f"{tag} resuming: {len(state['batches'])}/{len(batches)} batches done", flush=True)

    todo = [b for b in range(len(batches)) if str(b) not in state["batches"]]
    left = sum(len(batches[b]) for b in todo)
    t0, done_pairs = time.time(), 0
    for c in range(0, len(todo), CHUNK_BATCHES):
        for b in todo[c:c + CHUNK_BATCHES]:
            with Timer(device) as t:
                res = score_batch(model, [seqs[i] for i in batches[b]], device)
            state["batches"][str(b)] = {k: res[k] for k in KEYS}
            state["batch_ms"][str(b)] = t.ms
            done_pairs += len(batches[b])
        save_json(state, partial_path)
        rate = done_pairs / (time.time() - t0)
        print(f"{tag} {len(state['batches'])}/{len(batches)} batches  {rate:.2f} pairs/s  "
              f"ETA {(left - done_pairs) / rate / 60:.1f} min", flush=True)

    out = {k: [None] * len(seqs) for k in KEYS}
    for b, idx in enumerate(batches):
        r = state["batches"][str(b)]
        for k in KEYS:
            for j, i in enumerate(idx):
                out[k][i] = r[k][j]
    return out, [state["batch_ms"][str(b)] for b in range(len(batches))]


def _paired(x, y, qids, seed) -> dict:
    """x minus y, per query. W/L/T and bootstrap CI on nDCG@10."""
    dn = np.array([x["per_query"][q]["ndcg@10"] - y["per_query"][q]["ndcg@10"] for q in qids])
    dm = np.array([x["per_query"][q]["mrr@10"] - y["per_query"][q]["mrr@10"] for q in qids])
    rng = np.random.default_rng(seed)
    boot = dn[rng.integers(0, len(qids), size=(BOOT, len(qids)))].mean(1)
    return {
        "delta_ndcg@10": float(dn.mean()),
        "delta_mrr@10": float(dm.mean()),
        "ndcg@10_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "wlt_ndcg@10": [int((dn > EPS).sum()), int((dn < -EPS).sum()), int((np.abs(dn) <= EPS).sum())],
    }


def _agreement(s, ref, qids) -> dict:
    """Per-query Spearman of top-20 scores and top-1 agreement vs the reference."""
    rhos, top1 = [], 0
    for q in qids:
        docs = list(ref[q])
        a = np.array([ref[q][d] for d in docs])
        b = np.array([s[q][d] for d in docs])
        top1 += docs[int(a.argmax())] == docs[int(b.argmax())]
        rhos.append(float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1]))
    return {"spearman_mean": float(np.mean(rhos)), "top1_agree_frac": top1 / len(qids)}


def compute_metrics(ds, cand, ref, qids, var_scores, depth, seed, smoke) -> dict:
    cs = {q: cand["candidates"][q] for q in qids}
    qrels = {q: ds.qrels[q] for q in qids}
    qf = {q: ref["scores"][q] for q in qids}
    systems = {"bm25": identity_scores(cs, depth), "qwen3_qf": qf}
    systems.update({f"qwen3_df{v}": s for v, s in sorted(var_scores.items())})
    ev = {k: evaluate_rerank(cs, {q: s[q] for q in qids}, qrels, depth) for k, s in systems.items()}

    r100 = ev["bm25"]["mean"]["recall@100"]
    for k, e in ev.items():
        if abs(e["mean"]["recall@100"] - r100) > 1e-12:
            raise SystemExit(f"{k}: Recall@100 changed after reranking ({e['mean']['recall@100']} vs {r100})")

    out = {}
    for k, e in ev.items():
        entry = {"mean": e["mean"], "num_queries": e["num_queries"], "diagnostics": e["diagnostics"]}
        if k.startswith("qwen3_df"):
            p = _paired(e, ev["qwen3_qf"], qids, seed)
            entry.update({
                "per_query": e["per_query"],
                "vs_qf": p,
                "agreement_vs_qf": _agreement(systems[k], qf, qids),
                "order_gate_pass": None if smoke else bool(p["delta_ndcg@10"] >= -GATE),
            })
        out[k] = entry
    if "qwen3_dfA" in ev and "qwen3_dfB" in ev:
        out["B_vs_A"] = _paired(ev["qwen3_dfB"], ev["qwen3_dfA"], qids, seed)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--variants", nargs="+", default=["B", "A"], choices=["A", "B"],
                    help="variants to score if missing (scored in this order)")
    ap.add_argument("--smoke", type=int, default=0, help="score only the first N queries; writes under artifacts/smoke/")
    ap.add_argument("--force", action="store_true", help="rescore requested variants even if score files exist")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    depth, bs = cfg.rerank.depth, cfg.rerank.batch_size
    smoke = args.smoke > 0
    results, artifacts = cfg.paths.results_dir, cfg.paths.artifacts_dir
    score_dir = ensure_dir(artifacts / "smoke" / "scores" if smoke else results / "scores")
    partial_dir = ensure_dir(artifacts / ("smoke_partial" if smoke else "scores_partial"))
    metrics_path = (artifacts / "smoke" / "metrics" if smoke else results / "metrics") / "docfirst_zeroshot.json"
    ensure_dir(metrics_path.parent)
    device, dtype = get_device(cfg.runtime.device), get_dtype(cfg.runtime.dtype)
    info = run_info()
    model = None

    per_ds = {}
    for name in cfg.datasets:
        art = load_pt(artifacts / "tokenized" / f"{name}_qwen3.pt")
        dfart = load_pt(artifacts / "tokenized" / f"{name}_qwen3_df.pt")
        if dfart["base_content_sha256"] != art["content_sha256"]:
            raise SystemExit(f"[{name}] seam artifact was built from a different Phase 2 artifact. Rerun 02b.")
        qids = art["query_order"][:args.smoke] if smoke else art["query_order"]

        var_scores = {}
        for v in sorted({"A", "B"}, key=lambda x: args.variants.index(x) if x in args.variants else 9):
            tag = f"[{name} df{v}]"
            pairs, seqs = build_inputs(art, dfart, v, qids)
            in_sha = _sha(seqs)
            path = score_dir / f"qwen3_df{v}_{name}.json"
            requested = v in args.variants
            if path.exists() and not (args.force and requested):
                old = load_json(path)
                if old["inputs_sha256"] == in_sha and old["batch_size"] == bs:
                    print(f"{tag} reusing {path}")
                    var_scores[v] = old["scores"]
                    continue
                if requested:
                    raise SystemExit(f"{tag} {path} was scored on different inputs. Rerun with --force.")
            if not requested:
                continue

            if model is None:
                model = load_model(cfg.models.qwen3.name, cfg.models.qwen3.revision, device, dtype)
            meta = {"dataset": name, "variant": v, "inputs_sha256": in_sha, "num_pairs": len(seqs),
                    "batch_size": bs, "chunk_batches": CHUNK_BATCHES, "device": str(device),
                    "dtype": cfg.runtime.dtype}
            ppath = partial_dir / f"qwen3_df{v}_{name}.json"
            if args.force and ppath.exists():
                ppath.unlink()
            out, ms = score_run(model, seqs, bs, device, ppath, meta, tag)
            if not np.all(np.isfinite(out["log_odds"])):
                raise SystemExit(f"{tag} non-finite scores. Paste the log.")

            nest = {k: {} for k in KEYS}
            for (q, d), *vals in zip(pairs, *(out[k] for k in KEYS)):
                for k, x in zip(KEYS, vals):
                    nest[k].setdefault(q, {})[d] = round(float(x), 6)
            total_s = sum(ms) / 1000
            save_json({
                "run_info": info,
                "model": f"qwen3_df{v}",
                "name": cfg.models.qwen3.name,
                "revision": cfg.models.qwen3.revision,
                "dataset": name,
                "variant": v,
                "prompt_order": "doc_first",
                "num_pairs": len(seqs),
                "batch_size": bs,
                "chunk_batches": CHUNK_BATCHES,
                "device": str(device),
                "dtype": cfg.runtime.dtype,
                "inputs_sha256": in_sha,
                "depth": depth,
                "smoke_queries": args.smoke or None,
                "score_key": "log_odds",
                "aux_key": "p_yes",
                "input_stats": {"prompt_tokens": _len_stats([len(s) for s in seqs])},
                "timing": {"total_s": round(total_s, 2), "pairs_per_s": round(len(seqs) / total_s, 3),
                           "batch_ms": summarize_ms(ms)},
                "query_order": qids,
                "scores": nest["log_odds"],
                "aux": nest["p_yes"],
                "logit_yes": nest["logit_yes"],
                "logit_no": nest["logit_no"],
            }, path)
            ppath.unlink(missing_ok=True)
            print(f"{tag} saved {path}  ({len(seqs) / total_s:.2f} pairs/s)", flush=True)
            var_scores[v] = nest["log_odds"]

        if not var_scores:
            print(f"[{name}] no doc-first scores yet, skipping metrics")
            continue
        ds = load_beir(name, split=cfg.split, data_dir=cfg.paths.data_dir)
        cand = load_json(results / "candidates" / f"{name}.json")
        ref = load_json(results / "scores" / f"qwen3_{name}.json")
        per_ds[name] = compute_metrics(ds, cand, ref, qids, var_scores, depth, cfg.seed, smoke)

    save_json({"run_info": info, "depth": depth, "smoke_queries": args.smoke or None,
               "gate_threshold": GATE, "bootstrap_reps": BOOT, "datasets": per_ds}, metrics_path)

    print(f"\nmetrics -> {metrics_path}" + ("   (SMOKE: numbers are a pipeline check only)" if smoke else ""))
    print(f"{'dataset':9s} {'system':10s} {'nDCG@10':>8s} {'MRR@10':>7s} {'dNDCG':>8s} {'95% CI':>17s} "
          f"{'W/L/T':>11s} {'rho':>5s} {'top1':>5s} {'gate':>5s}")
    for name, m in per_ds.items():
        for k in ("bm25", "qwen3_qf", "qwen3_dfA", "qwen3_dfB"):
            if k not in m:
                continue
            e = m[k]
            row = f"{name:9s} {k:10s} {e['mean']['ndcg@10']:8.4f} {e['mean']['mrr@10']:7.4f}"
            if "vs_qf" in e:
                p, a = e["vs_qf"], e["agreement_vs_qf"]
                ci = f"[{p['ndcg@10_ci95'][0]:+.4f},{p['ndcg@10_ci95'][1]:+.4f}]"
                g = {None: "-", True: "PASS", False: "FAIL"}[e["order_gate_pass"]]
                row += (f" {p['delta_ndcg@10']:+8.4f} {ci:>17s} {'/'.join(map(str, p['wlt_ndcg@10'])):>11s}"
                        f" {a['spearman_mean']:5.2f} {a['top1_agree_frac']:5.2f} {g:>5s}")
            print(row)
        if "B_vs_A" in m:
            p = m["B_vs_A"]
            print(f"{name:9s} {'B - A':10s} {'':8s} {'':7s} {p['delta_ndcg@10']:+8.4f} "
                  f"[{p['ndcg@10_ci95'][0]:+.4f},{p['ndcg@10_ci95'][1]:+.4f}] "
                  f"{'/'.join(map(str, p['wlt_ndcg@10'])):>11s}")


if __name__ == "__main__":
    main()