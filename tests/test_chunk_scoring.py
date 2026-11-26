# tests/test_chunk_scoring.py
"""Phase 5.1: score_on_chunk (stacked chunk, two gathers per batch) == score_on_caches.

Default (CPU fp32): 6 SciFact docs (3 shortest, 3 longest of the first 300, so padding is real),
mixed-doc batches of 4. Max |d log-odds| <= 1e-5 (expected exactly 0: same tensors, shapes, ops),
identical rankings, chunk and DocKVs unchanged after scoring.
--mps: same check on MPS bf16 (expected exactly 0), then timing on real doc-major batches for both
datasets (old vs new, alternating which runs first); writes results/metrics/chunk_scoring.json.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys

import torch

from marginkv.config import load_config
from marginkv.kv import cache_ops as co
from marginkv.models import qwen3_reranker as qr
from marginkv.train.data import sha256_file
from marginkv.utils.device import get_dtype
from marginkv.utils.io import load_json, load_pt, run_info, save_json
from marginkv.utils.seed import set_seed
from marginkv.utils.timing import Timer, summarize_ms

TOL = 1e-5


def load_model(cfg, device, dtype_name):
    model_dir = cfg.paths.artifacts_dir / "lora" / "df" / "merged"
    shas = load_json(model_dir / "marginkv_merge.json")["files_sha256"]
    for fname, want in shas.items():
        assert sha256_file(model_dir / fname) == want, f"sha mismatch: {fname}"
    return qr.load_model(str(model_dir), None, device, get_dtype(dtype_name)), shas


def ds_inputs(cfg, ds):
    art = load_pt(cfg.paths.artifacts_dir / "tokenized" / f"{ds}_qwen3.pt")
    dfart = load_pt(cfg.paths.artifacts_dir / "tokenized" / f"{ds}_qwen3_df.pt")
    assert dfart["base_content_sha256"] == art["content_sha256"]
    pieces = {**art["pieces"], **dfart["seam_pieces"]}
    qbd = {}
    for qid in art["query_order"]:
        for d in art["rerank_candidates"][qid]:
            qbd.setdefault(d, []).append(qid)
    return art, dfart, pieces, co.shared_prefix_len(pieces), qbd


def build(model, pieces, dfart, docs, device, shared_len):
    out = {}
    for d in docs:
        prefix, _ = co.split_doc_first(pieces, dfart["doc_seam"][d], [], "B")
        out[d] = co.build_doc_kvs(model, [prefix], device, shared_len)[0]
    return out


def new_ids(pieces, dfart, art, d, q):
    return co.split_doc_first(pieces, dfart["doc_seam"][d], art["query_df"][q], "B")[1]


def check_exact(cfg, model, device):
    art, dfart, pieces, shared_len, qbd = ds_inputs(cfg, "scifact")
    cand = sorted((d for d in art["doc_order"][:300] if d in qbd), key=lambda d: dfart["cache_len_seam"][d])
    docs = cand[:3] + cand[-3:]
    kv_of = build(model, pieces, dfart, docs, device, shared_len)
    chunk = co.stack_chunk([kv_of[d] for d in docs])
    row = {d: i for i, d in enumerate(docs)}
    pairs = [(d, q) for d in docs for q in qbd[d]]
    pairs.sort(key=lambda p: (qbd[p[0]].index(p[1]), row[p[0]]))   # interleave docs: mixed lengths per batch

    snap_k, snap_v = chunk.keys.clone(), chunk.values.clone()
    snap_kv = {d: kv_of[d].clone() for d in docs}
    old_all, new_all = [], []
    for i in range(0, len(pairs), 4):
        b = pairs[i:i + 4]
        news = [new_ids(pieces, dfart, art, d, q) for d, q in b]
        old_all += co.score_on_caches(model, [kv_of[d] for d, _ in b], news, device)["log_odds"]
        new_all += co.score_on_chunk(model, chunk, [row[d] for d, _ in b], news, device)["log_odds"]

    max_diff = max(abs(a - b) for a, b in zip(old_all, new_all))
    rank_old = sorted(range(len(old_all)), key=lambda i: -old_all[i])
    rank_new = sorted(range(len(new_all)), key=lambda i: -new_all[i])
    assert max_diff <= TOL, f"chunk vs per-row max diff {max_diff:.3e} > {TOL}"
    assert rank_old == rank_new, "rankings differ"
    assert torch.equal(chunk.keys, snap_k) and torch.equal(chunk.values, snap_v), "chunk was modified"
    for d in docs:
        for a, b in zip(kv_of[d].keys + kv_of[d].values, snap_kv[d].keys + snap_kv[d].values):
            assert torch.equal(a, b), "DocKV was modified"
    lens = [kv_of[d].n_cached for d in docs]
    print(f"[exact {device}] {len(pairs)} pairs, cache lens {lens}, max diff {max_diff:.3e}, "
          f"exactly equal: {max_diff == 0.0}, rankings identical, no mutation  PASS")
    return {"pairs": len(pairs), "cache_lens": lens, "max_abs_diff": max_diff, "exact": max_diff == 0.0}


def time_paths(cfg, model, device, ds, n_docs=96, doc_chunk=16, score_batch=8, warmup=2):
    art, dfart, pieces, shared_len, qbd = ds_inputs(cfg, ds)
    docs = [d for d in art["doc_order"] if d in qbd][:n_docs]
    old_ms, new_ms, stack_ms, n_pairs, seen, max_diff = [], [], [], 0, 0, 0.0
    for c in range(0, len(docs), doc_chunk):
        cd = docs[c:c + doc_chunk]
        kv_of = build(model, pieces, dfart, cd, device, shared_len)
        with Timer(device) as ts:
            chunk = co.stack_chunk([kv_of[d] for d in cd])
        stack_ms.append(ts.ms)
        row = {d: i for i, d in enumerate(cd)}
        pairs = [(d, q) for d in cd for q in qbd[d]]
        for i in range(0, len(pairs), score_batch):
            b = pairs[i:i + score_batch]
            news = [new_ids(pieces, dfart, art, d, q) for d, q in b]
            kvs, rows = [kv_of[d] for d, _ in b], [row[d] for d, _ in b]
            if seen % 2 == 0:
                with Timer(device) as to:
                    old = co.score_on_caches(model, kvs, news, device)
                with Timer(device) as tn:
                    new = co.score_on_chunk(model, chunk, rows, news, device)
            else:
                with Timer(device) as tn:
                    new = co.score_on_chunk(model, chunk, rows, news, device)
                with Timer(device) as to:
                    old = co.score_on_caches(model, kvs, news, device)
            max_diff = max(max_diff, max(abs(a - b) for a, b in zip(old["log_odds"], new["log_odds"])))
            seen += 1
            if seen <= warmup:
                continue
            old_ms.append(to.ms)
            new_ms.append(tn.ms)
            n_pairs += len(b)
        del kv_of, chunk
    old_pp = sum(old_ms) / n_pairs
    new_pp = (sum(new_ms) + sum(stack_ms)) / n_pairs     # stacking cost charged to the new path
    r = {"batches": len(old_ms), "pairs": n_pairs, "old_ms": summarize_ms(old_ms), "new_ms": summarize_ms(new_ms),
         "stack_ms_per_chunk": summarize_ms(stack_ms), "old_ms_per_pair": old_pp,
         "new_ms_per_pair_incl_stack": new_pp, "speedup": old_pp / new_pp, "max_abs_diff": max_diff}
    print(f"[time {ds}] old {r['old_ms']['mean']:.1f} ms/batch, new {r['new_ms']['mean']:.1f} ms/batch, "
          f"stack {r['stack_ms_per_chunk']['mean']:.1f} ms/chunk | {old_pp:.1f} -> {new_pp:.1f} ms/pair "
          f"({r['speedup']:.2f}x), max diff {max_diff:.2e}")
    return r


def main():
    cfg = load_config()
    set_seed(cfg.seed)
    torch.set_grad_enabled(False)
    if "--mps" in sys.argv:
        device = torch.device("mps")
        model, shas = load_model(cfg, device, "bfloat16")
        res = {"run_info": run_info(), "phase": "5.1", "model_files_sha256": shas, "device": "mps", "dtype": "bfloat16",
               "settings": {"doc_chunk": 16, "score_batch": 8, "build_batch": 1, "n_docs": 96, "warmup": 2},
               "exact": check_exact(cfg, model, device), "timing": {}}
        for ds in cfg.datasets:
            res["timing"][ds] = time_paths(cfg, model, device, ds)
        save_json(res, cfg.paths.results_dir / "metrics" / "chunk_scoring.json")
    else:
        torch.set_num_threads(4)
        model, _ = load_model(cfg, torch.device("cpu"), "float32")
        check_exact(cfg, model, torch.device("cpu"))


if __name__ == "__main__":
    main()