# scripts/06d_equality_lora.py
"""Phase 4.5: restated equality gate on the doc-first LoRA (variant B), cached vs full prompt.

Model: artifacts/lora/df/merged (MPS bf16; CPU fp32 for the reference subset).
Full-prompt bf16 reference: qwen3_lora_dfB_{ds}.json from 06c (inputs asserted identical).
Cached bf16 run: doc KV built once per doc (build batch 1), tag + query + suffix on top (score batch 8).

Gate (restated 2026-11-25 in Phase 3.4; measurement fixed in 4.5 before data):
  strict : CPU fp32, cached vs full, max |d log-odds| <= 1e-4 on the subset
  tier2a : mean |cached bf16 - full fp32| <= mean |full bf16 - full fp32| on the pooled subset
  tier2b : |nDCG@10(cached bf16) - nDCG@10(full bf16)| <= 0.002 per dataset (all pairs)
Subset: first 16 queries x top 8 docs per dataset (256 pairs).
Resumable per doc chunk. --smoke 5 (matches the 06c smoke run).
Outputs: results/scores/qwen3_lora_dfB_cached_{ds}.json, results/metrics/equality_lora.json
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import gc
import hashlib
import json
import time

import numpy as np
import torch

from marginkv.config import load_config
from marginkv.data.beir_loader import load_beir
from marginkv.eval.rerank import evaluate_rerank
from marginkv.kv.cache_ops import build_doc_kvs, score_on_caches, shared_prefix_len, split_doc_first
from marginkv.models import qwen3_reranker as qr
from marginkv.train import data as td
from marginkv.utils.device import get_device, get_dtype
from marginkv.utils.io import ensure_dir, load_json, load_pt, run_info, save_json
from marginkv.utils.seed import set_seed
from marginkv.utils.timing import Timer, summarize_ms

KEYS = ("log_odds", "p_yes", "logit_yes", "logit_no")
NDCG_TOL, STRICT_TOL = 0.002, 1e-4
SUB_Q, SUB_D = 16, 8
SAVE_EVERY = 4
BOOT = 1000


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _dist(x) -> dict:
    x = np.asarray(x)
    return {"max": float(x.max()), "p99": float(np.percentile(x, 99)), "p50": float(np.percentile(x, 50)),
            "mean": float(x.mean())}


def _rank_stats(qids, top, a, b) -> dict:
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
    return {"queries_identical": same, "num_queries": len(qids), "pair_flips": flips,
            "pairs_compared": tot, "flip_rate": flips / tot}


def run_cached(model, device, chunks, by_doc, prefix, new, sl, build_batch, score_batch, ppath, meta, tag):
    """Same loop as 05_equality_test.run_cached."""
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
        for s in range(0, len(chunk), build_batch):
            sub = chunk[s:s + build_batch]
            with Timer(device) as t:
                built = build_doc_kvs(model, [prefix[d] for d in sub], device, sl)
            build_ms.append(t.ms)
            kvs.update(zip(sub, built))
        cp = [p for d in chunk for p in by_doc[d]]
        res, score_ms = {k: [] for k in KEYS}, []
        for s in range(0, len(cp), score_batch):
            bp = cp[s:s + score_batch]
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
    ap.add_argument("--smoke", type=int, default=0, help="must match the 06c smoke size (5)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--doc-chunk", type=int, default=16)
    ap.add_argument("--build-batch", type=int, default=1)
    ap.add_argument("--score-batch", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)
    depth = cfg.rerank.depth
    smoke = args.smoke > 0
    R, A = cfg.paths.results_dir, cfg.paths.artifacts_dir
    sdir = ensure_dir(A / "smoke" / "scores" if smoke else R / "scores")
    pdir = ensure_dir(A / ("smoke_partial" if smoke else "scores_partial"))
    metrics_path = ensure_dir(A / "smoke" / "metrics" if smoke else R / "metrics") / "equality_lora.json"
    device, dtype, cpu = get_device(cfg.runtime.device), get_dtype(cfg.runtime.dtype), torch.device("cpu")
    info = run_info()

    mdir = A / "lora" / "df" / "merged"
    merge_info = load_json(mdir / "marginkv_merge.json")
    files = {p: td.sha256_file(mdir / p) for p in merge_info["files_sha256"]}
    if files != merge_info["files_sha256"]:
        raise SystemExit("merged weights do not match marginkv_merge.json")
    settings = {"doc_chunk": args.doc_chunk, "build_batch": args.build_batch, "score_batch": args.score_batch}

    # ---- inputs per dataset
    D = {}
    for name in cfg.datasets:
        art = load_pt(A / "tokenized" / f"{name}_qwen3.pt")
        dfart = load_pt(A / "tokenized" / f"{name}_qwen3_df.pt")
        if dfart["base_content_sha256"] != art["content_sha256"]:
            raise SystemExit(f"[{name}] seam artifact does not match the Phase 2 artifact")
        qids = art["query_order"][:args.smoke] if smoke else art["query_order"]
        top, q_df = art["rerank_candidates"], art["query_df"]
        pb = {**art["pieces"], **dfart["seam_pieces"]}
        pairs = [(q, d) for q in qids for d in top[q]]
        docs = sorted({d for _, d in pairs})
        prefix = {d: split_doc_first(pb, dfart["doc_seam"][d], [], "B")[0] for d in docs}
        new = {p: split_doc_first(pb, dfart["doc_seam"][p[1]], q_df[p[0]], "B")[1] for p in pairs}
        ref = load_json(sdir / f"qwen3_lora_dfB_{name}.json")
        if _sha([prefix[d] + new[(q, d)] for q, d in pairs]) != ref["inputs_sha256"]:
            raise SystemExit(f"[{name}] inputs differ from qwen3_lora_dfB_{name}.json (smoke must be 5)")
        by_doc = {}
        for p in pairs:
            by_doc.setdefault(p[1], []).append(p)
        D[name] = dict(art=art, qids=qids, top=top, pairs=pairs, docs=docs, prefix=prefix, new=new, ref=ref,
                       by_doc=by_doc, sl=shared_prefix_len(pb),
                       chunks=[docs[i:i + args.doc_chunk] for i in range(0, len(docs), args.doc_chunk)],
                       sub=[(q, d) for q in qids[:SUB_Q] for d in top[q][:SUB_D]])

    # ---- cached bf16 run (MPS)
    model = None
    for name, x in D.items():
        tag = f"[{name} lora_dfB cached]"
        in_sha = _sha([x["pairs"], [x["prefix"][d] for d in x["docs"]], [x["new"][p] for p in x["pairs"]]])
        out_path = sdir / f"qwen3_lora_dfB_cached_{name}.json"
        if out_path.exists() and not args.force:
            old = load_json(out_path)
            if old["inputs_sha256"] == in_sha and old["settings"] == settings:
                print(f"{tag} reusing {out_path}")
                x["rec"] = old
                continue
            raise SystemExit(f"{tag} {out_path} has different inputs/settings. Rerun with --force.")
        if model is None:
            print(f"loading {mdir} on {device}", flush=True)
            model = qr.load_model(str(mdir), None, device, dtype)
        meta = {"dataset": name, "inputs_sha256": in_sha, **settings, "device": str(device),
                "dtype": cfg.runtime.dtype, "model_files_sha256": files}
        ppath = pdir / f"qwen3_lora_dfB_cached_{name}.json"
        if args.force and ppath.exists():
            ppath.unlink()
        st = run_cached(model, device, x["chunks"], x["by_doc"], x["prefix"], x["new"], x["sl"],
                        args.build_batch, args.score_batch, ppath, meta, tag)
        nest = {k: {} for k in KEYS}
        build_ms, score_ms, kv_bytes = [], [], 0
        for ci in range(len(x["chunks"])):
            c = st["chunks"][str(ci)]
            for i, (q, d) in enumerate(c["pairs"]):
                for k in KEYS:
                    nest[k].setdefault(q, {})[d] = round(float(c[k][i]), 6)
            build_ms += c["build_ms"]
            score_ms += c["score_ms"]
            kv_bytes += c["kv_bytes"]
        if not np.all(np.isfinite([nest["log_odds"][q][d] for q, d in x["pairs"]])):
            raise SystemExit(f"{tag} non-finite cached scores. Paste the log.")
        b_s, s_s = sum(build_ms) / 1000, sum(score_ms) / 1000
        full_rate = x["ref"]["timing"]["pairs_per_s"]
        n_p, n_d = len(x["pairs"]), len(x["docs"])
        rec = {
            "run_info": info, "model": "qwen3_lora_dfB_cached", "model_dir": "artifacts/lora/df/merged",
            "model_files_sha256": files, "dataset": name, "variant": "B", "device": str(device),
            "dtype": cfg.runtime.dtype, "settings": settings, "inputs_sha256": in_sha, "depth": depth,
            "smoke_queries": args.smoke or None, "num_pairs": n_p, "num_docs": n_d,
            "reuse_factor": round(n_p / n_d, 3),
            "timing": {
                "build_total_s": round(b_s, 2), "score_total_s": round(s_s, 2),
                "build_ms_per_doc": round(1000 * b_s / n_d, 2), "score_ms_per_pair": round(1000 * s_s / n_p, 2),
                "cached_pairs_per_s": round(n_p / (b_s + s_s), 3), "full_prompt_pairs_per_s": full_rate,
                "speedup_one_pass": round(n_p / (b_s + s_s) / full_rate, 3),
                "build_batch_ms": summarize_ms(build_ms), "score_batch_ms": summarize_ms(score_ms),
                "sweep_projection_min": {str(K): {"cached_min": round((b_s + K * s_s) / 60, 1),
                                                  "full_prompt_min": round(K * n_p / full_rate / 60, 1)}
                                         for K in (1, 4, 8, 16)},
            },
            "kv_bytes_per_doc_mean": round(kv_bytes / n_d), "query_order": x["qids"], "score_key": "log_odds",
            "scores": nest["log_odds"], "aux": nest["p_yes"],
            "logit_yes": nest["logit_yes"], "logit_no": nest["logit_no"],
        }
        save_json(rec, out_path)
        ppath.unlink(missing_ok=True)
        print(f"{tag} saved {out_path}", flush=True)
        x["rec"] = rec
    if model is not None:
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    # ---- CPU fp32 on the subset: full prompt and cached, one pair at a time
    print(f"cpu fp32 subset: {sum(len(x['sub']) for x in D.values())} pairs ...", flush=True)
    m32 = qr.load_model(str(mdir), None, cpu, torch.float32)
    for name, x in D.items():
        f32, c32 = [], []
        for q, d in x["sub"]:
            pre, nw = x["prefix"][d], x["new"][(q, d)]
            f32.append(qr.score_batch(m32, [pre + nw], cpu)["log_odds"][0])
            kv = build_doc_kvs(m32, [pre], cpu, x["sl"])[0]
            c32.append(score_on_caches(m32, [kv], [nw], cpu)["log_odds"][0])
        x["f32"], x["c32"] = np.array(f32, dtype=np.float64), np.array(c32, dtype=np.float64)
        print(f"[{name}] subset done", flush=True)
    del m32
    gc.collect()

    # ---- gate
    per_ds, e_full_all, e_cached_all, strict_all = {}, [], [], []
    for name, x in D.items():
        rec, ref = x["rec"], x["ref"]
        fb = np.array([ref["scores"][q][d] for q, d in x["sub"]])
        cb = np.array([rec["scores"][q][d] for q, d in x["sub"]])
        e_full, e_cached = np.abs(fb - x["f32"]), np.abs(cb - x["f32"])
        strict = np.abs(x["c32"] - x["f32"])
        e_full_all.append(e_full)
        e_cached_all.append(e_cached)
        strict_all.append(strict)

        ds = load_beir(name, split=cfg.split, data_dir=cfg.paths.data_dir)
        cand = load_json(R / "candidates" / f"{name}.json")
        cs = {q: cand["candidates"][q] for q in x["qids"]}
        qrels = {q: ds.qrels[q] for q in x["qids"]}
        ev_c = evaluate_rerank(cs, {q: rec["scores"][q] for q in x["qids"]}, qrels, depth)
        ev_f = evaluate_rerank(cs, {q: ref["scores"][q] for q in x["qids"]}, qrels, depth)
        d_ndcg = ev_c["mean"]["ndcg@10"] - ev_f["mean"]["ndcg@10"]
        d_all = np.abs(np.array([rec["scores"][q][d] - ref["scores"][q][d] for q, d in x["pairs"]]))
        per_ds[name] = {
            "num_pairs": len(x["pairs"]), "num_docs": len(x["docs"]), "subset_pairs": len(x["sub"]),
            "cached_vs_full_bf16_all": _dist(d_all),
            "ranking": _rank_stats(x["qids"], x["top"], rec["scores"], ref["scores"]),
            "cached": {"mean": ev_c["mean"]}, "full_prompt": {"mean": ev_f["mean"]},
            "delta_ndcg@10": d_ndcg, "delta_mrr@10": ev_c["mean"]["mrr@10"] - ev_f["mean"]["mrr@10"],
            "tier2b_pass": None if smoke else bool(abs(d_ndcg) <= NDCG_TOL),
            "subset": {"full_bf16_vs_fp32": _dist(e_full), "cached_bf16_vs_fp32": _dist(e_cached),
                       "fp32_cached_vs_full": _dist(strict)},
            "timing": rec["timing"], "kv_bytes_per_doc_mean": rec["kv_bytes_per_doc_mean"],
        }

    ef, ec, sa = np.concatenate(e_full_all), np.concatenate(e_cached_all), np.concatenate(strict_all)
    diff = ec - ef
    rng = np.random.default_rng(cfg.seed)
    boot = diff[rng.integers(0, len(diff), size=(BOOT, len(diff)))].mean(1)
    tier2a = {"pairs": len(diff), "mean_err_full_bf16": float(ef.mean()), "mean_err_cached_bf16": float(ec.mean()),
              "ratio": float(ec.mean() / ef.mean()), "diff_mean": float(diff.mean()),
              "diff_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
              "pass": None if smoke else bool(ec.mean() <= ef.mean())}
    strict = {"pairs": len(sa), **_dist(sa), "pass": bool(sa.max() <= STRICT_TOL)}
    t2b = [m["tier2b_pass"] for m in per_ds.values()]
    overall = None if smoke else bool(strict["pass"] and tier2a["pass"] and all(t2b))

    save_json({"run_info": info, "model_dir": "artifacts/lora/df/merged", "model_files_sha256": files,
               "settings": settings, "smoke_queries": args.smoke or None,
               "rule": {"strict": f"fp32 cached vs full max <= {STRICT_TOL}",
                        "tier2a": "mean |cached bf16 - fp32| <= mean |full bf16 - fp32| (pooled subset)",
                        "tier2b": f"|dNDCG@10 cached vs full bf16| <= {NDCG_TOL} per dataset",
                        "subset": f"first {SUB_Q} queries x top {SUB_D} docs per dataset"},
               "strict": strict, "tier2a": tier2a, "equality_gate_pass": overall, "datasets": per_ds}, metrics_path)

    print(f"\nmetrics -> {metrics_path}" + ("   (SMOKE)" if smoke else ""))
    print(f"strict (fp32 cached vs full, {strict['pairs']} pairs): max {strict['max']:.2e}  "
          f"{'PASS' if strict['pass'] else 'FAIL'}")
    t = tier2a
    g = {None: "-", True: "PASS", False: "FAIL"}[t["pass"]]
    print(f"tier2a (err vs fp32, {t['pairs']} pairs): full bf16 {t['mean_err_full_bf16']:.4f}  "
          f"cached bf16 {t['mean_err_cached_bf16']:.4f}  ratio {t['ratio']:.2f}  "
          f"diff CI [{t['diff_ci95'][0]:+.4f},{t['diff_ci95'][1]:+.4f}]  {g}")
    for name, m in per_ds.items():
        d, r = m["cached_vs_full_bf16_all"], m["ranking"]
        g = {None: "-", True: "PASS", False: "FAIL"}[m["tier2b_pass"]]
        tm = m["timing"]
        print(f"[{name}] tier2b dNDCG {m['delta_ndcg@10']:+.4f}  dMRR {m['delta_mrr@10']:+.4f}  {g} | "
              f"cached-vs-full max {d['max']:.3f} p99 {d['p99']:.3f} | flips {r['flip_rate']:.1e} "
              f"same {r['queries_identical']}/{r['num_queries']}")
        print(f"{'':10s} build {tm['build_ms_per_doc']:.0f} ms/doc  score {tm['score_ms_per_pair']:.0f} ms/pair  "
              f"KV {m['kv_bytes_per_doc_mean'] / 2**20:.1f} MiB/doc  speedup {tm['speedup_one_pass']:.2f}x")
    print(f"\nequality gate (restated), LoRA doc-first: {overall}")


if __name__ == "__main__":
    main()