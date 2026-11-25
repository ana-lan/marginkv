# scripts/03_score_reference.py
"""Phase 2.5: score the top-`depth` BM25 candidates with the reference rerankers and evaluate.

Models: qwen3 (query-first, log-odds of yes vs no) and bge (cross-encoder logit).
Outputs (all embed run_info()):
  results/scores/{model}_{ds}.json    {qid: {doc_id: score}} plus aux (P(yes) / sigmoid) and throughput
  results/metrics/reference.json      nDCG@10, MRR@10, Recall@100, diagnostics, deltas vs BM25
Resumable: progress is checkpointed per chunk under artifacts/scores_partial/.
--smoke N: first N queries only, everything written under artifacts/smoke/ (never results/).
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from marginkv.config import load_config
from marginkv.data.beir_loader import load_beir
from marginkv.eval.rerank import evaluate_rerank, identity_scores
from marginkv.models import bge_reranker as bge
from marginkv.models import qwen3_reranker as qwen
from marginkv.retrieval.bm25 import doc_text
from marginkv.utils.device import get_device, get_dtype
from marginkv.utils.io import ensure_dir, load_json, load_pt, run_info, save_json
from marginkv.utils.seed import set_seed
from marginkv.utils.timing import Timer

MODELS = ("qwen3", "bge")
SCORE_KEYS = {"qwen3": ("log_odds", "p_yes"), "bge": ("logit", "p")}
CHUNK_BATCHES = 16


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, separators=(",", ":")).encode()).hexdigest()


def _free(device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def build_inputs(model_key, cfg, name, ds, top, qids):
    """Returns pairs [(qid, doc_id)], token id sequences, and input stats."""
    pairs = [(q, d) for q in qids for d in top[q]]
    if model_key == "qwen3":
        a = load_pt(cfg.paths.artifacts_dir / "tokenized" / f"{name}_qwen3.pt")
        if a["model"]["revision"] != cfg.models.qwen3.revision or a["max_doc_tokens"] != cfg.rerank.max_doc_tokens:
            raise SystemExit(f"[{name}] tokenized artifact was built with a different model revision or max_doc_tokens")
        for q in qids:
            if a["rerank_candidates"][q] != top[q]:
                raise SystemExit(f"[{name}] tokenized rerank_candidates differ from candidates file for query {q}")
        seqs = [qwen.query_first_ids(a["pieces"], a["query_qf"][q], a["doc"][d]) for q, d in pairs]
        stats = {"tokenized_artifact_sha256": a["content_sha256"]}
        return pairs, seqs, stats

    tok = bge.load_tokenizer(cfg.models.bge.name, cfg.models.bge.revision)
    q_ids = dict(zip(qids, bge.encode(tok, [ds.queries[q] for q in qids])))
    docs = sorted({d for _, d in pairs})
    d_full = dict(zip(docs, bge.encode(tok, [doc_text(ds.corpus[d]) for d in docs])))
    m = cfg.rerank.max_doc_tokens
    seqs = [bge.pair_ids(tok, q_ids[q], d_full[d][:m]) for q, d in pairs]
    n_trunc = sum(len(v) > m for v in d_full.values())
    stats = {"num_unique_docs": len(docs), "docs_truncated": n_trunc,
             "docs_truncated_frac": round(n_trunc / len(docs), 4), "pad_id": tok.pad_token_id}
    return pairs, seqs, stats


def score_all(model_key, model, seqs, batch_size, device, partial_path, meta, pad_id=None, tag=""):
    """Chunked, resumable scoring. Returns (score list, aux list, timing dict) in input order."""
    n = len(seqs)
    order = sorted(range(n), key=lambda i: (len(seqs[i]), i))
    size = batch_size * CHUNK_BATCHES
    chunks = [order[i:i + size] for i in range(0, n, size)]
    key, aux_key = SCORE_KEYS[model_key]

    state = {"meta": meta, "done": 0, "score": {}, "aux": {}, "chunk_ms": [], "chunk_tokens": []}
    if partial_path.exists():
        old = load_json(partial_path)
        if old["meta"] != meta:
            raise SystemExit(f"{partial_path} was made with different inputs/settings. Delete it to start over.")
        state = old
        print(f"{tag} resuming at chunk {state['done'] + 1}/{len(chunks)}")

    for ci in range(state["done"], len(chunks)):
        idx = chunks[ci]
        sub = [seqs[i] for i in idx]
        with Timer(device) as t:
            if model_key == "qwen3":
                res = qwen.score_sequences(model, sub, batch_size, device)
            else:
                res = bge.score_sequences(model, sub, pad_id, batch_size, device)
        for j, i in enumerate(idx):
            state["score"][str(i)] = res[key][j]
            state["aux"][str(i)] = res[aux_key][j]
        state["done"] = ci + 1
        state["chunk_ms"].append(t.ms)
        state["chunk_tokens"].append(sum(len(s) for s in sub))
        save_json(state, partial_path)
        done_pairs = sum(len(c) for c in chunks[:ci + 1])
        rate = len(idx) / (t.ms / 1000)
        left = (n - done_pairs) / rate if rate > 0 else float("nan")
        print(f"{tag} chunk {ci + 1}/{len(chunks)}  {done_pairs}/{n} pairs  {rate:5.1f} pairs/s  "
              f"len<={len(sub[-1])}  eta {left / 60:4.1f} min")

    ms, toks = state["chunk_ms"], state["chunk_tokens"]
    steady_ms, steady_tok = (ms[1:], toks[1:]) if len(ms) > 1 else (ms, toks)
    steady_pairs = n - len(chunks[0]) if len(ms) > 1 else n
    timing = {
        "pairs": n, "batch_size": batch_size, "chunks": len(chunks),
        "total_s": round(sum(ms) / 1000, 2),
        "pairs_per_s_total": round(n / (sum(ms) / 1000), 2),
        "pairs_per_s_steady": round(steady_pairs / (sum(steady_ms) / 1000), 2),
        "tokens_per_s_steady": round(sum(steady_tok) / (sum(steady_ms) / 1000), 1),
        "note": "steady excludes chunk 1 (warmup). Chunk timings may span resumed sessions.",
    }
    scores = [state["score"][str(i)] for i in range(n)]
    aux = [state["aux"][str(i)] for i in range(n)]
    return scores, aux, timing


def _vs_bm25(m, bm25_m, metric="ndcg@10"):
    pq, bq = m["per_query"], bm25_m["per_query"]
    first = next(iter(pq.values()))
    assert isinstance(first, dict) and metric in first, f"unexpected per_query shape: {type(first)}"
    w = sum(pq[q][metric] > bq[q][metric] + 1e-12 for q in pq)
    l = sum(pq[q][metric] < bq[q][metric] - 1e-12 for q in pq)
    return {"metric": metric, "wins": w, "losses": l, "ties": len(pq) - w - l}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="default: cfg.rerank.batch_size")
    ap.add_argument("--smoke", type=int, default=0, help="first N queries only, outputs under artifacts/smoke/")
    ap.add_argument("--force", action="store_true", help="re-score even if a final scores file exists")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    depth = cfg.rerank.depth
    bs = args.batch_size or cfg.rerank.batch_size
    device, dtype = get_device(cfg.runtime.device), get_dtype(cfg.runtime.dtype)
    datasets = args.datasets or list(cfg.datasets)
    smoke = args.smoke > 0
    out_root = cfg.paths.artifacts_dir / "smoke" if smoke else cfg.paths.results_dir
    partial_root = cfg.paths.artifacts_dir / ("smoke_partial" if smoke else "scores_partial")
    info = run_info()
    print(f"device={device} dtype={dtype} batch_size={bs} depth={depth} smoke={args.smoke or 'off'}")

    metrics_path = out_root / "metrics" / "reference.json"
    report = load_json(metrics_path) if metrics_path.exists() else {"datasets": {}}
    report.update({"run_info": info, "depth": depth, "pool_size": cfg.retrieval.pool_size,
                   "smoke_queries": args.smoke or None})
    summary = []

    for name in datasets:
        ds = load_beir(name, split=cfg.split, data_dir=cfg.paths.data_dir)
        cfile = load_json(cfg.paths.results_dir / "candidates" / f"{name}.json")
        qids = cfile["query_order"][:args.smoke] if smoke else cfile["query_order"]
        cands = {q: cfile["candidates"][q] for q in qids}
        qrels = {q: ds.qrels[q] for q in qids}
        top = {q: [d for d, _ in cands[q][:depth]] for q in qids}
        bm25_m = evaluate_rerank(cands, identity_scores(cands, depth), qrels, depth)
        dsr = report["datasets"].setdefault(name, {})
        dsr["bm25"] = {"mean": bm25_m["mean"], "num_queries": bm25_m["num_queries"]}

        for mk in args.models:
            tag = f"[{mk}/{name}]"
            spec = getattr(cfg.models, mk)
            score_path = out_root / "scores" / f"{mk}_{name}.json"
            if score_path.exists() and not args.force:
                print(f"{tag} scores exist, evaluating only: {score_path}")
            else:
                pairs, seqs, in_stats = build_inputs(mk, cfg, name, ds, top, qids)
                meta = {"model": mk, "name": spec.name, "revision": spec.revision, "dataset": name,
                        "num_pairs": len(seqs), "batch_size": bs, "chunk_batches": CHUNK_BATCHES,
                        "device": str(device), "dtype": str(dtype), "inputs_sha256": _sha(seqs)}
                if mk == "qwen3":
                    model = qwen.load_model(spec.name, spec.revision, device, dtype)
                    pad = None
                else:
                    model = bge.load_model(spec.name, spec.revision, device, dtype)
                    pad = in_stats["pad_id"]
                partial = ensure_dir(partial_root) / f"{mk}_{name}.json"
                scores, aux, timing = score_all(mk, model, seqs, bs, device, partial, meta, pad, tag)
                del model
                _free(device)

                key, aux_key = SCORE_KEYS[mk]
                bad = [i for i, s in enumerate(scores) if not torch.isfinite(torch.tensor(s))]
                if bad:
                    raise SystemExit(f"{tag} {len(bad)} non-finite scores, e.g. pair {pairs[bad[0]]}")
                sc, ax = {}, {}
                for (q, d), s, a_ in zip(pairs, scores, aux):
                    sc.setdefault(q, {})[d] = round(s, 6)
                    ax.setdefault(q, {})[d] = round(a_, 6)
                save_json({
                    "run_info": info, **meta, "depth": depth, "score_key": key, "aux_key": aux_key,
                    "input_stats": in_stats, "timing": timing,
                    "query_order": qids, "scores": sc, "aux": ax,
                }, score_path)
                partial.unlink()
                print(f"{tag} saved {score_path}")

            # Evaluate what is on disk.
            saved = load_json(score_path)
            m = evaluate_rerank(cands, saved["scores"], qrels, depth)
            r100 = abs(m["mean"]["recall@100"] - bm25_m["mean"]["recall@100"])
            assert r100 < 1e-12, f"{tag} Recall@100 changed by {r100}: reranking must only reorder"
            dsr[mk] = {
                "name": saved["name"], "revision": saved["revision"], "score_key": saved["score_key"],
                "device": saved["device"], "dtype": saved["dtype"], "batch_size": saved["batch_size"],
                "mean": m["mean"], "num_queries": m["num_queries"], "per_query": m["per_query"],
                "delta_vs_bm25": {k: round(m["mean"][k] - bm25_m["mean"][k], 4) for k in ("ndcg@10", "mrr@10")},
                "vs_bm25": _vs_bm25(m, bm25_m),
                "diagnostics": m["diagnostics"], "timing": saved["timing"], "input_stats": saved["input_stats"],
            }
            summary.append((name, mk, dsr[mk]))

    save_json(report, metrics_path)
    print(f"\nsaved {metrics_path}\n")
    print(f"{'dataset':9s} {'model':6s} {'nDCG@10':>8s} {'dNDCG':>7s} {'MRR@10':>7s} {'dMRR':>7s} "
          f"{'R@100':>7s} {'W/L/T':>11s} {'tieQ':>5s} {'shift':>6s} {'pairs/s':>8s}")
    for name, mk, r in summary:
        b = report["datasets"][name]["bm25"]["mean"]
        print(f"{name:9s} {'bm25':6s} {b['ndcg@10']:8.4f} {'':>7s} {b['mrr@10']:7.4f} {'':>7s} {b['recall@100']:7.4f}") \
            if mk == summary[0][1] or summary[[s[0] for s in summary].index(name)][1] == mk else None
        v = r["vs_bm25"]
        print(f"{name:9s} {mk:6s} {r['mean']['ndcg@10']:8.4f} {r['delta_vs_bm25']['ndcg@10']:+7.4f} "
              f"{r['mean']['mrr@10']:7.4f} {r['delta_vs_bm25']['mrr@10']:+7.4f} {r['mean']['recall@100']:7.4f} "
              f"{v['wins']:>3d}/{v['losses']:>3d}/{v['ties']:>3d} {r['diagnostics']['queries_with_score_ties']:5d} "
              f"{r['diagnostics']['mean_abs_rank_shift']:6.2f} {r['timing']['pairs_per_s_steady']:8.1f}")


if __name__ == "__main__":
    main()