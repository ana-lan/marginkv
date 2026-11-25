# scripts/05_equality_test.py
"""Phase 3.4: equality gate at scale + cached-path throughput.

Scores every reranked pair on the cached doc-first path (doc KV built once per unique doc, then
tag + query + suffix on top) and compares with the full-prompt doc-first scores from
04_docfirst_zeroshot.py. Same model, device and dtype as that reference.

Practical tier of the equality gate (MPS bf16), per dataset:
    max |d log-odds| <= 0.2  and  |d nDCG@10| <= 0.002      (ranking flips reported)
The strict tier (CPU fp32) is tests/test_cache_equality_small.py.

Noise reference: a subset of full prompts re-scored one at a time vs the saved batched run,
i.e. the reference's own bf16 batch-composition noise, for reading the cached-vs-full numbers.

Docs are streamed in chunks (all doc KVs at once would be ~190 GB). Resumable per chunk. --smoke N.
Outputs (full run):
  results/scores/qwen3_df{V}_cached_{ds}.json
  results/metrics/equality.json
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # must be set before transformers is imported

import argparse
import hashlib
import json
import time

import numpy as np
import torch

from marginkv.config import load_config
from marginkv.data.beir_loader import load_beir
from marginkv.eval.rerank import evaluate_rerank
from marginkv.kv.cache_ops import build_doc_kvs, score_on_caches, shared_prefix_len, split_doc_first
from marginkv.models.qwen3_reranker import load_model, score_batch
from marginkv.utils.device import get_device, get_dtype
from marginkv.utils.io import ensure_dir, load_json, load_pt, run_info, save_json
from marginkv.utils.seed import set_seed
from marginkv.utils.timing import Timer, summarize_ms

KEYS = ("log_odds", "p_yes", "logit_yes", "logit_no")
MAX_TOL, NDCG_TOL = 0.2, 0.002
SAVE_EVERY = 4  # chunks


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _dist(x) -> dict:
    x = np.asarray(x)
    return {"max": float(x.max()), "p99": float(np.percentile(x, 99)), "p50": float(np.percentile(x, 50)),
            "mean": float(x.mean())}


def _rank_stats(qids, top, a, b) -> dict:
    """Queries with identical top-depth order, and pairwise order flips (ties in either ignored)."""
    same = flips = tot = 0
    for q in qids:
        docs = top[q]
        x = np.array([a[q][d] for d in docs])
        y = np.array([b[q][d] for d in docs])
        same += np.argsort(-x, kind="stable").tolist() == np.argsort(-y, kind="stable").tolist()
        iu = np.triu_indices(len(docs), 1)
        sx = np.sign(x[:, None] - x[None, :])[iu]
        sy = np.sign(y[:, None] - y[None, :])[iu]
        flips += int((sx * sy < 0).sum())
        tot += len(iu[0])
    return {"queries_identical": same, "num_queries": len(qids), "pair_flips": flips, "pairs_compared": tot,
            "flip_rate": flips / tot}


def run_cached(model, device, chunks, by_doc, prefix, new, sl, args, ppath, meta, tag):
    state = {"meta": meta, "chunks": {}}
    if ppath.exists():
        old = load_json(ppath)
        if old.get("meta") != meta:
            raise SystemExit(f"{tag} partial {ppath} has different inputs/settings. Delete it or use --force.")
        state = old
        print(f"{tag} resuming: {len(state['chunks'])}/{len(chunks)} chunks done", flush=True)

    todo = [ci for ci in range(len(chunks)) if str(ci) not in state["chunks"]]
    left = sum(len(by_doc[d]) for ci in todo for d in chunks[ci])
    t0, done = time.time(), 0
    for n, ci in enumerate(todo):
        chunk = chunks[ci]
        kvs, build_ms = {}, []
        for s in range(0, len(chunk), args.build_batch):
            sub = chunk[s:s + args.build_batch]
            with Timer(device) as t:
                built = build_doc_kvs(model, [prefix[d] for d in sub], device, sl)
            build_ms.append(t.ms)
            kvs.update(zip(sub, built))

        cp = [p for d in chunk for p in by_doc[d]]
        res, score_ms = {k: [] for k in KEYS}, []
        for s in range(0, len(cp), args.score_batch):
            bp = cp[s:s + args.score_batch]
            with Timer(device) as t:
                r = score_on_caches(model, [kvs[d] for _, d in bp], [new[p] for p in bp], device)
            score_ms.append(t.ms)
            for k in KEYS:
                res[k] += r[k]

        state["chunks"][str(ci)] = {"pairs": [list(p) for p in cp], **res, "build_ms": build_ms,
                                    "score_ms": score_ms, "kv_bytes": sum(kv.nbytes() for kv in kvs.values())}
        del kvs
        if device.type == "mps":
            torch.mps.empty_cache()
        done += len(cp)
        if (n + 1) % SAVE_EVERY == 0 or n == len(todo) - 1:
            save_json(state, ppath)
            rate = done / (time.time() - t0)
            print(f"{tag} {len(state['chunks'])}/{len(chunks)} chunks  {rate:.2f} pairs/s (build+score)  "
                  f"ETA {(left - done) / rate / 60:.1f} min", flush=True)
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--variant", default="B", choices=["A", "B"])
    ap.add_argument("--smoke", type=int, default=0, help="first N queries; reads/writes under artifacts/smoke/")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--doc-chunk", type=int, default=16)
    ap.add_argument("--build-batch", type=int, default=1)  # MPS bf16 padded builds shift KV ~2% rel (3.4 diag)
    ap.add_argument("--score-batch", type=int, default=16)
    ap.add_argument("--noise-pairs", type=int, default=256)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    V, depth = args.variant, cfg.rerank.depth
    smoke = args.smoke > 0
    results, artifacts = cfg.paths.results_dir, cfg.paths.artifacts_dir
    ref_dir = artifacts / "smoke" / "scores" if smoke else results / "scores"
    out_dir = ensure_dir(ref_dir)
    partial_dir = ensure_dir(artifacts / ("smoke_partial" if smoke else "scores_partial"))
    metrics_path = (artifacts / "smoke" / "metrics" if smoke else results / "metrics") / "equality.json"
    ensure_dir(metrics_path.parent)
    device, dtype = get_device(cfg.runtime.device), get_dtype(cfg.runtime.dtype)
    info = run_info()
    model = load_model(cfg.models.qwen3.name, cfg.models.qwen3.revision, device, dtype)

    per_ds = {}
    for name in cfg.datasets:
        tag = f"[{name} df{V} cached]"
        art = load_pt(artifacts / "tokenized" / f"{name}_qwen3.pt")
        dfart = load_pt(artifacts / "tokenized" / f"{name}_qwen3_df.pt")
        if dfart["base_content_sha256"] != art["content_sha256"]:
            raise SystemExit(f"[{name}] seam artifact was built from a different Phase 2 artifact.")
        qids = art["query_order"][:args.smoke] if smoke else art["query_order"]
        top, q_df = art["rerank_candidates"], art["query_df"]
        pieces = art["pieces"] if V == "A" else {**art["pieces"], **dfart["seam_pieces"]}
        docmap = art["doc"] if V == "A" else dfart["doc_seam"]
        sl = shared_prefix_len(pieces)

        pairs = [(q, d) for q in qids for d in top[q]]
        docs = sorted({d for _, d in pairs})
        prefix = {d: split_doc_first(pieces, docmap[d], [], V)[0] for d in docs}
        new = {p: split_doc_first(pieces, docmap[p[1]], q_df[p[0]], V)[1] for p in pairs}
        by_doc = {}
        for p in pairs:
            by_doc.setdefault(p[1], []).append(p)
        chunks = [docs[i:i + args.doc_chunk] for i in range(0, len(docs), args.doc_chunk)]

        ref_path = ref_dir / f"qwen3_df{V}_{name}.json"
        if not ref_path.exists():
            raise SystemExit(f"{tag} missing reference {ref_path}. Run 04_docfirst_zeroshot.py first.")
        ref = load_json(ref_path)
        missing = [p for p in pairs if p[0] not in ref["scores"] or p[1] not in ref["scores"][p[0]]]
        if missing:
            raise SystemExit(f"{tag} reference lacks {len(missing)} pairs (smoke size mismatch?): {missing[:3]}")

        in_sha = _sha([pairs, [prefix[d] for d in docs], [new[p] for p in pairs]])
        settings = {"doc_chunk": args.doc_chunk, "build_batch": args.build_batch, "score_batch": args.score_batch}
        out_path = out_dir / f"qwen3_df{V}_cached_{name}.json"
        rec = None
        if out_path.exists() and not args.force:
            old = load_json(out_path)
            if old["inputs_sha256"] == in_sha and old["settings"] == settings:
                print(f"{tag} reusing {out_path}")
                rec = old
            else:
                raise SystemExit(f"{tag} {out_path} has different inputs/settings. Rerun with --force.")

        if rec is None:
            meta = {"dataset": name, "variant": V, "inputs_sha256": in_sha, **settings,
                    "device": str(device), "dtype": cfg.runtime.dtype}
            ppath = partial_dir / f"qwen3_df{V}_cached_{name}.json"
            if args.force and ppath.exists():
                ppath.unlink()
            state = run_cached(model, device, chunks, by_doc, prefix, new, sl, args, ppath, meta, tag)

            nest = {k: {} for k in KEYS}
            build_ms, score_ms, kv_bytes = [], [], 0
            for ci in range(len(chunks)):
                c = state["chunks"][str(ci)]
                for i, (q, d) in enumerate(c["pairs"]):
                    for k in KEYS:
                        nest[k].setdefault(q, {})[d] = round(float(c[k][i]), 6)
                build_ms += c["build_ms"]
                score_ms += c["score_ms"]
                kv_bytes += c["kv_bytes"]
            vals = [nest["log_odds"][q][d] for q, d in pairs]
            if not np.all(np.isfinite(vals)):
                raise SystemExit(f"{tag} non-finite cached scores. Paste the log.")

            b_s, s_s = sum(build_ms) / 1000, sum(score_ms) / 1000
            full_rate = ref["timing"]["pairs_per_s"]
            sweep = {}
            for K in (1, 4, 8, 16):
                sweep[str(K)] = {"cached_min": round((b_s + K * s_s) / 60, 1),
                                 "full_prompt_min": round(K * len(pairs) / full_rate / 60, 1)}
            rec = {
                "run_info": info, "model": f"qwen3_df{V}_cached", "name": cfg.models.qwen3.name,
                "revision": cfg.models.qwen3.revision, "dataset": name, "variant": V,
                "device": str(device), "dtype": cfg.runtime.dtype, "settings": settings,
                "inputs_sha256": in_sha, "depth": depth, "smoke_queries": args.smoke or None,
                "num_pairs": len(pairs), "num_docs": len(docs), "reuse_factor": round(len(pairs) / len(docs), 3),
                "timing": {
                    "build_total_s": round(b_s, 2), "score_total_s": round(s_s, 2),
                    "build_ms_per_doc": round(1000 * b_s / len(docs), 2),
                    "score_ms_per_pair": round(1000 * s_s / len(pairs), 2),
                    "cached_pairs_per_s": round(len(pairs) / (b_s + s_s), 3),
                    "full_prompt_pairs_per_s": full_rate,
                    "speedup_one_pass": round(len(pairs) / (b_s + s_s) / full_rate, 3),
                    "build_batch_ms": summarize_ms(build_ms), "score_batch_ms": summarize_ms(score_ms),
                    "sweep_projection_min": sweep,
                },
                "kv_bytes_per_doc_mean": round(kv_bytes / len(docs)),
                "query_order": qids, "score_key": "log_odds",
                "scores": nest["log_odds"], "aux": nest["p_yes"],
                "logit_yes": nest["logit_yes"], "logit_no": nest["logit_no"],
            }
            save_json(rec, out_path)
            ppath.unlink(missing_ok=True)
            print(f"{tag} saved {out_path}", flush=True)

        # Noise reference: full prompts scored one at a time vs the saved batched full-prompt run.
        n_noise = min(args.noise_pairs, len(pairs))
        sub = pairs[::max(1, len(pairs) // n_noise)][:n_noise]
        single = [score_batch(model, [prefix[d] + new[(q, d)]], device)["log_odds"][0] for q, d in sub]
        noise = np.abs(np.array(single) - np.array([ref["scores"][q][d] for q, d in sub]))

        # Cached vs full-prompt comparison.
        c_lo = np.array([rec["scores"][q][d] for q, d in pairs])
        r_lo = np.array([ref["scores"][q][d] for q, d in pairs])
        d_lo = np.abs(c_lo - r_lo)
        d_yes = np.abs(np.array([rec["logit_yes"][q][d] - ref["logit_yes"][q][d] for q, d in pairs]))
        d_no = np.abs(np.array([rec["logit_no"][q][d] - ref["logit_no"][q][d] for q, d in pairs]))

        ds = load_beir(name, split=cfg.split, data_dir=cfg.paths.data_dir)
        cand = load_json(results / "candidates" / f"{name}.json")
        cs = {q: cand["candidates"][q] for q in qids}
        qrels = {q: ds.qrels[q] for q in qids}
        ev_c = evaluate_rerank(cs, {q: rec["scores"][q] for q in qids}, qrels, depth)
        ev_r = evaluate_rerank(cs, {q: ref["scores"][q] for q in qids}, qrels, depth)
        d_ndcg = ev_c["mean"]["ndcg@10"] - ev_r["mean"]["ndcg@10"]
        d_mrr = ev_c["mean"]["mrr@10"] - ev_r["mean"]["mrr@10"]
        gate = None if smoke else bool(d_lo.max() <= MAX_TOL and abs(d_ndcg) <= NDCG_TOL)

        per_ds[name] = {
            "num_pairs": len(pairs), "num_docs": len(docs),
            "abs_diff_log_odds": _dist(d_lo), "abs_diff_logit_yes": _dist(d_yes), "abs_diff_logit_no": _dist(d_no),
            "noise_ref": {"pairs": n_noise, **_dist(noise)},
            "ranking": _rank_stats(qids, top, rec["scores"], ref["scores"]),
            "cached": {"mean": ev_c["mean"]}, "full_prompt": {"mean": ev_r["mean"]},
            "delta_ndcg@10": d_ndcg, "delta_mrr@10": d_mrr,
            "equality_gate_pass": gate, "timing": rec["timing"], "kv_bytes_per_doc_mean": rec["kv_bytes_per_doc_mean"],
        }

    save_json({"run_info": info, "variant": V, "smoke_queries": args.smoke or None,
               "tolerances": {"max_abs_log_odds": MAX_TOL, "abs_delta_ndcg@10": NDCG_TOL},
               "datasets": per_ds}, metrics_path)

    print(f"\nmetrics -> {metrics_path}" + ("   (SMOKE)" if smoke else ""))
    print(f"{'dataset':9s} {'pairs':>6s} {'max|d|':>7s} {'p99':>7s} {'mean':>8s} {'noiseMax':>8s} {'noiseP99':>8s} "
          f"{'dNDCG':>8s} {'dMRR':>8s} {'flips':>9s} {'sameQ':>8s} {'gate':>5s}")
    for name, m in per_ds.items():
        d, n, r = m["abs_diff_log_odds"], m["noise_ref"], m["ranking"]
        g = {None: "-", True: "PASS", False: "FAIL"}[m["equality_gate_pass"]]
        print(f"{name:9s} {m['num_pairs']:6d} {d['max']:7.4f} {d['p99']:7.4f} {d['mean']:8.5f} "
              f"{n['max']:8.4f} {n['p99']:8.4f} {m['delta_ndcg@10']:+8.4f} {m['delta_mrr@10']:+8.4f} "
              f"{r['flip_rate']:9.2e} {r['queries_identical']:>3d}/{r['num_queries']:<4d} {g:>5s}")
    print()
    for name, m in per_ds.items():
        t = m["timing"]
        print(f"{name:9s} docs {m['num_docs']}  build {t['build_ms_per_doc']:.1f} ms/doc  "
              f"score {t['score_ms_per_pair']:.1f} ms/pair  cached {t['cached_pairs_per_s']:.2f} pairs/s  "
              f"vs full {t['full_prompt_pairs_per_s']:.2f} ({t['speedup_one_pass']:.2f}x)  "
              f"KV {m['kv_bytes_per_doc_mean'] / 2**20:.1f} MiB/doc")
        print(f"{'':9s} sweep projection (min), cached vs full prompt: " +
              "  ".join(f"K={k}: {v['cached_min']} vs {v['full_prompt_min']}" for k, v in t["sweep_projection_min"].items()))


if __name__ == "__main__":
    main()