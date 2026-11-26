# scripts/07p_profile_cached.py
"""Phase 5.1: where does one cached score batch spend its time?

Times the three parts of score_on_caches on real doc-first LoRA batches:
  (a) pad: zero-alloc + copy each DocKV into [B, 8, P, 128] per layer
  (b) wrap: DynamicCache() + cache.update per layer
  (c) forward: mask/ids/positions + backbone forward on the new tokens + fp32 yes/no head
Also runs the real score_on_caches on the same batches (end-to-end time and a score-equality check).

Pre-registered rule (5.1): if (a) + (b) < 10% of batch time, keep score_on_caches as is and skip
pre-padding. Otherwise build a pre-padded chunk path and test it against the current one.
Docs are built one at a time (MPS padding note). Batching mirrors the sweep: chunks of docs,
all of a chunk's pairs in query order, cut into score batches.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse

import torch
from transformers import DynamicCache

from marginkv.config import load_config
from marginkv.kv import cache_ops as co
from marginkv.models import qwen3_reranker as qr
from marginkv.models.qwen3_reranker import PAD_ID
from marginkv.train.data import sha256_file
from marginkv.utils.device import get_device, get_dtype
from marginkv.utils.io import load_json, load_pt, run_info, save_json
from marginkv.utils.seed import set_seed
from marginkv.utils.timing import Timer, summarize_ms

THRESHOLD = 0.10


def verify_model(model_dir):
    meta = load_json(model_dir / "marginkv_merge.json")
    shas = meta["files_sha256"]
    if not isinstance(shas, dict):
        raise SystemExit(f"files_sha256 is {type(shas).__name__}, expected {{file: sha}}; paste marginkv_merge.json")
    for fname, want in shas.items():
        got = sha256_file(model_dir / fname)
        if got != want:
            raise SystemExit(f"sha mismatch for {fname}: {got} != {want}")
    return shas


@torch.inference_mode()
def profile_batch(model, kvs, new_seqs, device):
    """Same ops as co.score_on_caches, split into timed parts."""
    B = len(kvs)
    P = max(kv.n_cached for kv in kvs)
    N = max(len(s) for s in new_seqs)
    ref = kvs[0].keys[0]
    H, D, dt = ref.shape[0], ref.shape[-1], ref.dtype
    L = kvs[0].num_layers

    with Timer(device) as t_pad:
        ks, vs = [], []
        for layer in range(L):
            k = torch.zeros(B, H, P, D, dtype=dt, device=device)
            v = torch.zeros(B, H, P, D, dtype=dt, device=device)
            for b, kv in enumerate(kvs):
                n = kv.n_cached
                k[b, :, P - n:] = kv.keys[layer]
                v[b, :, P - n:] = kv.values[layer]
            ks.append(k)
            vs.append(v)

    with Timer(device) as t_wrap:
        cache = DynamicCache()
        for layer in range(L):
            cache.update(ks[layer], vs[layer], layer)

    with Timer(device) as t_fwd:
        ids = torch.full((B, N), PAD_ID, dtype=torch.long)
        mask = torch.zeros((B, P + N), dtype=torch.long)
        pos = torch.zeros((B, N), dtype=torch.long)
        for b, (kv, s) in enumerate(zip(kvs, new_seqs)):
            m = len(s)
            ids[b, N - m:] = torch.tensor(s, dtype=torch.long)
            mask[b, P - kv.n_cached:P] = 1
            mask[b, P + N - m:] = 1
            pos[b, N - m:] = torch.arange(kv.next_pos, kv.next_pos + m)
        out = model.model(input_ids=ids.to(device), attention_mask=mask.to(device),
                          position_ids=pos.to(device), past_key_values=cache, use_cache=True)
        res = co._yes_no(model, out.last_hidden_state[:, -1, :])

    del ks, vs, cache, out
    return t_pad.ms, t_wrap.ms, t_fwd.ms, res["log_odds"], {"B": B, "P": P, "N": N}


def batches_for(art, dfart, pieces, n_batches, doc_chunk, score_batch):
    """Yield (chunk_docs, [(doc_id, prefix, new), ...] per batch) until n_batches batches."""
    qids_by_doc = {}
    for qid in art["query_order"]:
        for d in art["rerank_candidates"][qid]:
            qids_by_doc.setdefault(d, []).append(qid)
    docs = [d for d in art["doc_order"] if d in qids_by_doc]
    made = 0
    for c in range(0, len(docs), doc_chunk):
        chunk = docs[c:c + doc_chunk]
        pairs = []
        for d in chunk:
            for qid in qids_by_doc[d]:
                prefix, new = co.split_doc_first(pieces, dfart["doc_seam"][d], art["query_df"][qid], "B")
                pairs.append((d, prefix, new))
        # keep pair order = doc order then query order (like a doc-major sweep)
        batches = [pairs[i:i + score_batch] for i in range(0, len(pairs), score_batch)]
        yield chunk, batches
        made += len(batches)
        if made >= n_batches:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=40, help="timed score batches per dataset")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--doc-chunk", type=int, default=16)
    ap.add_argument("--score-batch", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)
    torch.set_grad_enabled(False)
    device = get_device(cfg.runtime.device)
    dtype = get_dtype(cfg.runtime.dtype)

    model_dir = cfg.paths.artifacts_dir / "lora" / "df" / "merged"
    shas = verify_model(model_dir)
    model = qr.load_model(str(model_dir), None, device, dtype)

    out = {"run_info": run_info(), "phase": "5.1", "model_dir": str(model_dir), "model_files_sha256": shas,
           "device": str(device), "dtype": cfg.runtime.dtype, "threshold": THRESHOLD,
           "settings": {"batches": args.batches, "warmup": args.warmup, "doc_chunk": args.doc_chunk,
                        "score_batch": args.score_batch, "build_batch": 1},
           "datasets": {}}

    for ds in cfg.datasets:
        art = load_pt(cfg.paths.artifacts_dir / "tokenized" / f"{ds}_qwen3.pt")
        dfart = load_pt(cfg.paths.artifacts_dir / "tokenized" / f"{ds}_qwen3_df.pt")
        assert dfart["base_content_sha256"] == art["content_sha256"], "seam artifact not built from this Phase 2 artifact"
        pieces = {**art["pieces"], **dfart["seam_pieces"]}
        shared_len = co.shared_prefix_len(pieces)

        pad_ms, wrap_ms, fwd_ms, e2e_ms, build_ms, shapes = [], [], [], [], [], []
        max_diff = 0.0
        seen = 0
        for chunk, batches in batches_for(art, dfart, pieces, args.batches + args.warmup,
                                          args.doc_chunk, args.score_batch):
            kv_of = {}
            for d in chunk:
                prefix, _ = co.split_doc_first(pieces, dfart["doc_seam"][d], [], "B")
                with Timer(device) as tb:
                    kv_of[d] = co.build_doc_kvs(model, [prefix], device, shared_len)[0]
                build_ms.append(tb.ms)
            for batch in batches:
                kvs = [kv_of[d] for d, _, _ in batch]
                news = [n for _, _, n in batch]
                p, w, f, lo, shp = profile_batch(model, kvs, news, device)
                with Timer(device) as te:
                    ref = co.score_on_caches(model, kvs, news, device)
                diff = max(abs(a - b) for a, b in zip(lo, ref["log_odds"]))
                seen += 1
                if seen <= args.warmup:
                    continue
                if seen > args.warmup + args.batches:
                    break
                pad_ms.append(p); wrap_ms.append(w); fwd_ms.append(f); e2e_ms.append(te.ms)
                shapes.append(shp)
                max_diff = max(max_diff, diff)
            del kv_of
            if seen >= args.warmup + args.batches:
                break

        tot = [a + b + c for a, b, c in zip(pad_ms, wrap_ms, fwd_ms)]
        mean = lambda xs: sum(xs) / len(xs)
        overhead = (mean(pad_ms) + mean(wrap_ms)) / mean(tot)
        out["datasets"][ds] = {
            "timed_batches": len(tot),
            "pad_ms": summarize_ms(pad_ms), "wrap_ms": summarize_ms(wrap_ms), "forward_ms": summarize_ms(fwd_ms),
            "parts_total_ms": summarize_ms(tot), "score_on_caches_ms": summarize_ms(e2e_ms),
            "build_ms_per_doc": summarize_ms(build_ms),
            "overhead_frac": overhead,
            "ms_per_pair": mean(e2e_ms) / mean([s["B"] for s in shapes]),
            "shape_mean": {k: mean([s[k] for s in shapes]) for k in ("B", "P", "N")},
            "split_vs_real_max_abs_log_odds": max_diff,
        }
        r = out["datasets"][ds]
        print(f"[{ds}] pad {r['pad_ms']['mean']:.1f} ms  wrap {r['wrap_ms']['mean']:.1f} ms  "
              f"forward {r['forward_ms']['mean']:.1f} ms  real {r['score_on_caches_ms']['mean']:.1f} ms  "
              f"overhead {overhead:.1%}  {r['ms_per_pair']:.1f} ms/pair  split-vs-real {max_diff:.2e}")

    worst = max(d["overhead_frac"] for d in out["datasets"].values())
    out["worst_overhead_frac"] = worst
    out["decision"] = "skip_prepad" if worst < THRESHOLD else "build_prepad"
    print(f"worst overhead {worst:.1%} -> {out['decision']}")
    save_json(out, cfg.paths.results_dir / "metrics" / "cached_profile.json")


if __name__ == "__main__":
    main()