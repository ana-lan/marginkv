# src/marginkv/kv/sweep.py
"""Doc-major cached scoring sweep (Phases 5 to 7).

Build each doc's KV once (build batch 1), then for every config apply its KV transform and score all
of that doc chunk's pairs with score_on_caches, cut into batches of the config's score_batch.
Pair order is fixed: docs in doc_order (sorted ids), each doc's queries in query_order. Every config
sees the same pairs in the same batches (for a given score_batch), so configs differ only in the KV.

configs: {name: (transform, score_batch)}, transform(DocKV) -> DocKV (must not modify its input).
state is plain lists/dicts so it can be saved with save_pt and resumed per chunk.
"""
from __future__ import annotations

import math
from typing import Callable

from marginkv.kv import cache_ops as co
from marginkv.utils.timing import Timer

Transform = Callable[[co.DocKV], co.DocKV]


def identity(kv: co.DocKV) -> co.DocKV:
    return kv


def sweep_plan(art: dict, dfart: dict, qids=None):
    """(variant B pieces, docs in sweep order, {doc: [qid, ...] in query order})."""
    pieces = {**art["pieces"], **dfart["seam_pieces"]}
    qset = set(art["query_order"] if qids is None else qids)
    by_doc: dict[str, list[str]] = {}
    for qid in art["query_order"]:
        if qid in qset:
            for d in art["rerank_candidates"][qid]:
                by_doc.setdefault(d, []).append(qid)
    docs = [d for d in art["doc_order"] if d in by_doc]
    return pieces, docs, by_doc


def new_state(configs: dict) -> dict:
    return {"done_chunks": 0, "build_ms": [], "out": {n: {} for n in configs},
            "score_ms": {n: [] for n in configs}, "extra": {}}


def run_sweep(model, art, dfart, device, configs: dict, *, qids=None, doc_chunk: int = 16,
              state: dict | None = None, on_chunk=None, on_doc=None) -> dict:
    """Run (or resume) the sweep. out[name][qid][doc] = [log_odds, logit_yes, logit_no, p_yes].
    on_chunk(state, done, total) runs after each chunk (save partials there).
    on_doc(state, name, doc_id, base_kv, kv) runs once per doc and config (5.4 recon error)."""
    pieces, docs, by_doc = sweep_plan(art, dfart, qids)
    shared_len = co.shared_prefix_len(pieces)
    state = new_state(configs) if state is None else state
    if set(state["out"]) != set(configs):
        raise ValueError("state configs differ from requested configs")
    total = math.ceil(len(docs) / doc_chunk)

    for ci in range(state["done_chunks"], total):
        cd = docs[ci * doc_chunk:(ci + 1) * doc_chunk]
        base = {}
        for d in cd:
            prefix, _ = co.split_doc_first(pieces, dfart["doc_seam"][d], [], "B")
            with Timer(device) as t:
                base[d] = co.build_doc_kvs(model, [prefix], device, shared_len)[0]
            state["build_ms"].append(t.ms)
        pairs = [(d, q) for d in cd for q in by_doc[d]]
        news = [co.split_doc_first(pieces, dfart["doc_seam"][d], art["query_df"][q], "B")[1] for d, q in pairs]

        for name, (fn, sb) in configs.items():
            kvs = {d: fn(base[d]) for d in cd}
            if on_doc is not None:
                for d in cd:
                    on_doc(state, name, d, base[d], kvs[d])
            out = state["out"][name]
            for i in range(0, len(pairs), sb):
                bp, bn = pairs[i:i + sb], news[i:i + sb]
                with Timer(device) as t:
                    r = co.score_on_caches(model, [kvs[d] for d, _ in bp], bn, device)
                state["score_ms"][name].append(t.ms)
                for j, (d, q) in enumerate(bp):
                    out.setdefault(q, {})[d] = [r["log_odds"][j], r["logit_yes"][j], r["logit_no"][j], r["p_yes"][j]]
            del kvs
        del base
        state["done_chunks"] = ci + 1
        if on_chunk is not None:
            on_chunk(state, ci + 1, total)
    return state