# scripts/11a_build_traces.py
"""Phase 8a.1: build vLLM request traces 1 to 3 from frozen artifacts, plus trace 3
document repeat rates and a block-level LRU prefix-cache simulation. Mac only, no model.

Traces (pairs of [qid, doc_id], in request order):
  t1: same query, 20 docs. Queries with pairwise disjoint top-20 sets (by token content),
      query-major. Best case for query-first, no cross-query reuse for doc-first.
  t2: same doc, 20 queries. Distinct-content docs, each with 20 seeded queries, doc-major.
      Best case for doc-first.
  t3: the real BEIR stream: queries in file order, each with its 20 rerank candidates
      in BM25 rank order. These are exactly the eval pairs.

Outputs:
  results/traces/{ds}_{t}.json        pair lists (committed)
  artifacts/traces/{ds}_{t}.pt        prompt token ids for both orders (for the pod)
  results/metrics/trace_stats.json    repeat stats + simulated prefix-cache hits
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import random
from collections import Counter, OrderedDict

import numpy as np

from marginkv.config import load_config
from marginkv.models import qwen3_reranker as qr
from marginkv.train.data import content_sha256
from marginkv.utils.io import ensure_dir, load_json, load_pt, run_info, save_json, save_pt

BLOCK = 16
TRACES = ("t1", "t2", "t3")
DESCRIPTIONS = {
    "t1": "same query, 20 docs; queries with disjoint top-20 sets (by content); query-major",
    "t2": "same doc, 20 seeded queries; distinct-content docs; doc-major",
    "t3": "real BEIR stream: queries in file order x 20 rerank candidates in BM25 rank order",
}
# Estimated 4090 capacity in 16-token blocks (about 18 GiB of KV at 1.75 MiB per BF16 block).
# Replaced by the measured num_gpu_blocks in 8a.6.
DEFAULT_CAPS = "2500,5000,10500,21000,inf"


# ---------- loading and prompts ----------

def load_inputs(cfg, ds):
    tok_dir = cfg.paths.artifacts_dir / "tokenized"
    art = load_pt(tok_dir / f"{ds}_qwen3.pt")
    dfart = load_pt(tok_dir / f"{ds}_qwen3_df.pt")
    assert dfart["base_content_sha256"] == art["content_sha256"], "seam artifact does not match base"
    pieces = {**art["pieces"], **dfart["seam_pieces"]}
    cands = load_json(cfg.paths.results_dir / "candidates" / f"{ds}.json")["candidates"]
    zero_q = {q for q, lst in cands.items() if lst[0][1] == 0}
    return art, dfart, pieces, zero_q


def content_key(art, doc):
    return tuple(art["doc"][doc])


def build_prompts(art, dfart, pieces, pairs):
    """Token ids for both orders, plus each request's doc token span [start, end)."""
    pre_qf = len(pieces["prefix"]) + len(pieces["qf_head"])
    shared = len(pieces["prefix"]) + len(pieces["df_head"])
    head = pieces["prefix"] + pieces["df_head"]
    qf_ids, df_ids, qf_span, df_span, cache_len = [], [], [], [], []
    for qid, doc in pairs:
        q_qf, q_df = art["query_qf"][qid], art["query_df"][qid]
        d, d_seam = art["doc"][doc], dfart["doc_seam"][doc]
        a = qr.query_first_ids(pieces, q_qf, d)
        b = qr.doc_first_seam_ids(pieces, d_seam, q_df)
        cl = dfart["cache_len_seam"][doc]
        assert cl == shared + len(d_seam)
        assert b[:cl] == head + d_seam, f"doc-first prefix mismatch for {doc}"
        start = pre_qf + len(q_qf) + len(pieces["qf_mid"])
        assert a[start:start + len(d)] == d
        qf_ids.append(a)
        df_ids.append(b)
        qf_span.append((start, start + len(d)))
        df_span.append((shared, cl))
        cache_len.append(cl)
    return {"qf_ids": qf_ids, "df_ids": df_ids, "qf_span": qf_span, "df_span": df_span,
            "df_cache_len": cache_len, "shared_len": shared, "qf_shared_len": pre_qf}


# ---------- traces ----------

def trace3(art):
    return [[q, d] for q in art["query_order"] for d in art["rerank_candidates"][q]]


def trace1(art, seed, n_queries, per):
    rng = random.Random(f"{seed}-t1")
    order = list(art["query_order"])
    rng.shuffle(order)
    used, chosen = set(), []
    for q in order:
        keys = [content_key(art, d) for d in art["rerank_candidates"][q][:per]]
        if len(set(keys)) == len(keys) and not (set(keys) & used):
            chosen.append(q)
            used.update(keys)
            if len(chosen) == n_queries:
                break
    pos = {q: i for i, q in enumerate(art["query_order"])}
    chosen.sort(key=pos.get)
    pairs = [[q, d] for q in chosen for d in art["rerank_candidates"][q][:per]]
    return pairs, {"queries_found": len(chosen), "queries_wanted": n_queries}


def trace2(art, seed, n_docs, per):
    rng = random.Random(f"{seed}-t2")
    order = list(art["doc_order"])
    rng.shuffle(order)
    seen, docs = set(), []
    for d in order:
        k = content_key(art, d)
        if k not in seen:
            seen.add(k)
            docs.append(d)
            if len(docs) == n_docs:
                break
    pairs = []
    for d in docs:
        qs = random.Random(f"{seed}-t2-{d}").sample(list(art["query_order"]), per)
        pairs.extend([q, d] for q in qs)
    return pairs, {"docs_found": len(docs), "docs_wanted": n_docs}


# ---------- stats ----------

def pct(xs, p):
    return float(np.percentile(xs, p)) if xs else None


def repeat_stats(art, pairs, zero_q):
    n = len(pairs)
    ids = [d for _, d in pairs]
    keys = [content_key(art, d) for d in ids]
    uid, ukey = len(set(ids)), len(set(keys))

    occ = Counter(ids)
    buckets = Counter()
    for c in occ.values():
        b = "1" if c == 1 else "2" if c == 2 else "3-5" if c <= 5 else "6-10" if c <= 10 else "11+"
        buckets[b] += 1

    last, dist = {}, []
    for i, k in enumerate(keys):
        if k in last:
            dist.append(i - last[k])
        last[k] = i

    within_q, cur_q, cur = 0, None, set()
    for (q, _), k in zip(pairs, keys):
        if q != cur_q:
            cur_q, cur = q, set()
        if k in cur:
            within_q += 1
        cur.add(k)

    kept = [(q, d) for q, d in pairs if q not in zero_q]
    k_ids = [d for _, d in kept]
    k_keys = [content_key(art, d) for d in k_ids]
    excl = None
    if kept and len(kept) < n:
        excl = {"zero_overlap_queries": len({q for q, _ in pairs if q in zero_q}),
                "n_requests": len(kept),
                "repeat_frac_id": 1 - len(set(k_ids)) / len(kept),
                "repeat_frac_content": 1 - len(set(k_keys)) / len(kept)}

    return {
        "n_requests": n,
        "n_queries": len({q for q, _ in pairs}),
        "unique_docs_id": uid,
        "unique_docs_content": ukey,
        "repeat_frac_id": 1 - uid / n,
        "repeat_frac_content": 1 - ukey / n,
        "reuse_factor_id": n / uid,
        "occurrences_per_doc_id": dict(sorted(buckets.items())),
        "max_occurrences_id": max(occ.values()),
        "reuse_distance_requests": {"n": len(dist), "p50": pct(dist, 50), "p90": pct(dist, 90),
                                    "max": max(dist) if dist else None},
        "within_query_content_repeats": within_q,
        "excluding_zero_overlap": excl,
    }


def block_hashes(ids):
    out, h = [], None
    for k in range(len(ids) // BLOCK):
        h = hash((h, tuple(ids[k * BLOCK:(k + 1) * BLOCK])))
        out.append(h)
    return out


def simulate(prompts, spans, cap):
    """Sequential block-level prefix cache, LRU over cached full blocks.
    Hit = longest run of cached leading blocks, capped at (n - 1) // BLOCK.
    On completion, blocks are touched deepest first so a prefix's first blocks are evicted last."""
    cache = OrderedDict()
    hit_tok = tot_tok = doc_hit = doc_tot = evictions = req_hit = 0
    for ids, (a, b) in zip(prompts, spans):
        n = len(ids)
        hs = block_hashes(ids)
        max_hit = (n - 1) // BLOCK
        hit = 0
        while hit < max_hit and hs[hit] in cache:
            hit += 1
        ht = hit * BLOCK
        hit_tok += ht
        tot_tok += n
        doc_hit += max(0, min(ht, b) - a)
        doc_tot += b - a
        req_hit += ht > 0
        for h in reversed(hs):
            cache[h] = True
            cache.move_to_end(h)
        if cap is not None:
            while len(cache) > cap:
                cache.popitem(last=False)
                evictions += 1
    return {"hit_tokens": hit_tok, "prompt_tokens": tot_tok, "hit_frac": hit_tok / tot_tok,
            "doc_tokens_reused_frac": doc_hit / doc_tot, "requests_with_hit_frac": req_hit / len(prompts),
            "evictions": evictions, "cached_blocks_end": len(cache)}


def len_stats(xs):
    return {"p50": pct(xs, 50), "p95": pct(xs, 95), "max": max(xs), "mean": float(np.mean(xs))}


# ---------- main ----------

def guard_write_json(obj, path, force):
    if path.exists() and not force:
        old = load_json(path)
        if old.get("content_sha256") != obj["content_sha256"]:
            raise SystemExit(f"{path} exists with different pairs; use --force to overwrite")
    save_json(obj, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--t1-queries", type=int, default=50)
    ap.add_argument("--t2-docs", type=int, default=50)
    ap.add_argument("--per", type=int, default=20)
    ap.add_argument("--caps", default=DEFAULT_CAPS, help="LRU capacities in 16-token blocks; 'inf' = unbounded")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    datasets = args.datasets or list(cfg.datasets)
    caps = [None if c == "inf" else int(c) for c in args.caps.split(",")]
    cap_name = lambda c: "inf" if c is None else str(c)

    trace_dir = ensure_dir(cfg.paths.results_dir / "traces")
    art_dir = ensure_dir(cfg.paths.artifacts_dir / "traces")
    out = {"run_info": run_info(), "phase": "8a.1",
           "rules": {"block_size": BLOCK, "max_hit": "(n - 1) // block_size blocks",
                     "eviction": "LRU over cached full blocks, deepest block touched first",
                     "simulation": "sequential (concurrency 1), ignores blocks held by in-flight requests",
                     "caps_blocks": [cap_name(c) for c in caps],
                     "caps_note": "estimated for a 4090 (about 10,500 BF16 / 21,000 FP8 blocks); rerun with measured num_gpu_blocks"},
           "settings": {"seed": cfg.seed, "t1_queries": args.t1_queries, "t2_docs": args.t2_docs, "per": args.per},
           "datasets": {}}

    for ds in datasets:
        art, dfart, pieces, zero_q = load_inputs(cfg, ds)
        print(f"\n=== {ds}: {len(art['query_order'])} queries, {len(art['doc_order'])} docs, "
              f"zero-overlap queries {len(zero_q)}")
        built = {"t3": (trace3(art), {}),
                 "t1": trace1(art, cfg.seed, args.t1_queries, args.per),
                 "t2": trace2(art, cfg.seed, args.t2_docs, args.per)}
        ds_out = {"zero_overlap_queries": len(zero_q), "traces": {}}

        for t in TRACES:
            pairs, sel = built[t]
            sha = content_sha256(pairs)
            guard_write_json({"run_info": run_info(), "dataset": ds, "split": art["split"], "trace": t,
                              "description": DESCRIPTIONS[t], "seed": cfg.seed, "selection": sel,
                              "base_content_sha256": art["content_sha256"],
                              "df_content_sha256": dfart["content_sha256"],
                              "n_requests": len(pairs), "content_sha256": sha, "pairs": pairs},
                             trace_dir / f"{ds}_{t}.json", args.force)

            pr = build_prompts(art, dfart, pieces, pairs)
            save_pt({"dataset": ds, "trace": t, "pairs": pairs, "pairs_sha256": sha,
                     "model": art["model"], "variant": "B", **pr}, art_dir / f"{ds}_{t}.pt")

            st = repeat_stats(art, pairs, zero_q)
            sim = {order: {cap_name(c): simulate(pr[f"{order}_ids"], pr[f"{order}_span"], c) for c in caps}
                   for order in ("qf", "df")}
            ds_out["traces"][t] = {
                "description": DESCRIPTIONS[t], "selection": sel, "pairs_sha256": sha,
                "prompt_len": {"qf": len_stats([len(x) for x in pr["qf_ids"]]),
                               "df": len_stats([len(x) for x in pr["df_ids"]])},
                "df_cache_len": len_stats(pr["df_cache_len"]),
                "df_block_aligned_doc_frac": float(np.mean(
                    [max(0, (cl // BLOCK) * BLOCK - pr["shared_len"]) / (cl - pr["shared_len"])
                     for cl in pr["df_cache_len"]])),
                "repeats": st, "sim": sim}

            print(f"[{t}] {len(pairs)} requests {sel or ''} | repeat id {st['repeat_frac_id']:.3f} "
                  f"content {st['repeat_frac_content']:.3f} | reuse dist p50 {st['reuse_distance_requests']['p50']}")
            for order in ("qf", "df"):
                row = "  ".join(f"{k}:{v['hit_frac']:.3f}/{v['doc_tokens_reused_frac']:.3f}"
                                for k, v in sim[order].items())
                print(f"    sim {order} hit/doc-reused  {row}")
            if st["excluding_zero_overlap"]:
                e = st["excluding_zero_overlap"]
                print(f"    excl. zero-overlap: repeat id {e['repeat_frac_id']:.3f} content {e['repeat_frac_content']:.3f}")
        out["datasets"][ds] = ds_out

    save_json(out, cfg.paths.results_dir / "metrics" / "trace_stats.json")
    print("\nwrote results/metrics/trace_stats.json")


if __name__ == "__main__":
    main()