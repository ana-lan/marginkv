# tests/test_cache_equality_small.py
"""Phase 3.3: cached doc-first scoring == full-prompt doc-first scoring (variant B, SciFact).

Default: CPU fp32, strict tier of the equality gate (<= 1e-4 log-odds, identical rankings).
KV tensor comparisons use per-tensor relative error (Qwen3 K/V reach |x| ~ 1e2 in some layers).
Shared-prefix tolerance is set against storage precision: the shared KV will be stored once in
bf16 or lower (rel rounding ~4e-3), so fp32 differences far below that don't matter.
--mps adds the MPS bf16 check (run it after the long MPS job finishes).
Run: python tests/test_cache_equality_small.py [--mps]
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
from collections import Counter

import numpy as np
import torch

from marginkv.config import load_config
from marginkv.kv.cache_ops import build_doc_kvs, score_on_caches, shared_prefix_len, split_doc_first
from marginkv.models.qwen3_reranker import load_model, score_sequences
from marginkv.utils.device import get_device, get_dtype
from marginkv.utils.io import load_pt

STRICT_TOL = 1e-4       # log-odds, strict tier
BF16_TOL = 0.2          # log-odds, bf16 tier
KV_REL_TOL = 1e-5       # build padding invariance, fp32
SHARED_REL_TOL = 1e-4   # shared prefix across docs: ~40x below bf16 rounding


def load_inputs(cfg, name="scifact"):
    t = cfg.paths.artifacts_dir / "tokenized"
    art, dfart = load_pt(t / f"{name}_qwen3.pt"), load_pt(t / f"{name}_qwen3_df.pt")
    assert dfart["base_content_sha256"] == art["content_sha256"]
    pb = {**art["pieces"], **dfart["seam_pieces"]}
    top, qo = art["rerank_candidates"], art["query_order"]
    # Docs that several queries share (real reuse), plus 8 docs of one query (ranking check).
    cnt = Counter(d for q in qo for d in top[q])
    multi = [d for d in art["doc_order"] if cnt[d] >= 3][:3]
    pairs = []
    for d in multi:
        pairs += [(q, d) for q in qo if d in top[q]][:4]
    pairs += [(qo[0], d) for d in top[qo[0]][:8] if (qo[0], d) not in pairs]
    return art, dfart, pb, pairs


def kv_diff(a, b, upto=None):
    """(max abs diff, max per-tensor relative diff, where) over all layers' keys and values.
    Relative = max|x - y| / max|y| per tensor. where = (kind, layer, token) of the worst relative diff."""
    s = slice(None) if upto is None else slice(0, upto)
    L = len(a.keys)
    ab, rel, where = 0.0, 0.0, None
    for i, (x, y) in enumerate(zip([*a.keys, *a.values], [*b.keys, *b.values])):
        x, y = x[:, s].float(), y[:, s].float()
        diff = (x - y).abs()
        d = float(diff.max())
        r = d / max(float(y.abs().max()), 1e-12)
        ab = max(ab, d)
        if r > rel:
            tok = int(diff.amax(dim=(0, 2)).argmax())   # [heads, tokens, dim] -> worst token
            rel, where = r, ("keys" if i < L else "values", i % L, tok)
    return ab, rel, where


def rank_flips(pairs, a, b):
    byq = {}
    for (q, d), x, y in zip(pairs, a, b):
        byq.setdefault(q, []).append((d, x, y))
    checked = flips = 0
    for rows in byq.values():
        if len(rows) < 2:
            continue
        checked += 1
        flips += [r[0] for r in sorted(rows, key=lambda r: -r[1])] != [r[0] for r in sorted(rows, key=lambda r: -r[2])]
    return checked, flips


def run(cfg, device, dtype_name, strict):
    tol = STRICT_TOL if strict else BF16_TOL
    art, dfart, pb, pairs = load_inputs(cfg)
    q_df, doc_seam = art["query_df"], dfart["doc_seam"]
    model = load_model(cfg.models.qwen3.name, cfg.models.qwen3.revision, device, get_dtype(dtype_name))
    label = f"{device.type} {dtype_name}"

    split = {p: split_doc_first(pb, doc_seam[p[1]], q_df[p[0]], "B") for p in pairs}
    ref = np.array(score_sequences(model, [split[p][0] + split[p][1] for p in pairs], 8, device)["log_odds"])

    docs = sorted({d for _, d in pairs})
    sl = shared_prefix_len(pb)
    pre = {d: split_doc_first(pb, doc_seam[d], [], "B")[0] for d in docs}
    kvs = dict(zip(docs, build_doc_kvs(model, [pre[d] for d in docs], device, sl)))

    # 1. Positions: cache length and next position equal the 3.1 artifact's cache_len_seam.
    for d in docs:
        assert kvs[d].n_cached == kvs[d].next_pos == dfart["cache_len_seam"][d], d

    # 2. Build padding invariance, on the most-padded doc (shortest prefix in the batch).
    d_short = min(docs, key=lambda d: len(pre[d]))
    pad_amt = max(len(p) for p in pre.values()) - len(pre[d_short])
    alone = build_doc_kvs(model, [pre[d_short]], device, sl)[0]
    pad_abs, pad_rel, _ = kv_diff(alone, kvs[d_short])

    # 3. Shared prefix (system + instruction) KV across docs.
    sh = [kv_diff(kvs[d], kvs[docs[0]], upto=sl) for d in docs[1:]]
    sh_abs = max(x[0] for x in sh)
    sh_rel, sh_where = max(((x[1], x[2]) for x in sh), key=lambda t: t[0])

    # 4. Cached scoring, batched with mixed docs, and one pair at a time.
    news = [split[p][1] for p in pairs]
    got_b = []
    for s in range(0, len(pairs), 8):
        got_b += score_on_caches(model, [kvs[d] for _, d in pairs[s:s + 8]], news[s:s + 8], device)["log_odds"]
    got_b = np.array(got_b)
    got_s = np.array([score_on_caches(model, [kvs[d]], [news[i]], device)["log_odds"][0]
                      for i, (_, d) in enumerate(pairs)])

    # 5. Immutability: scoring again gives the same result and leaves the DocKV untouched.
    d0 = pairs[0][1]
    snap = kvs[d0].clone()
    again = np.array(score_on_caches(model, [kvs[d] for _, d in pairs[:8]], news[:8], device)["log_odds"])
    unchanged = kv_diff(snap, kvs[d0])[0] == 0.0
    repeat = float(np.abs(again - got_b[:8]).max())

    dq_b, dq_s = float(np.abs(got_b - ref).max()), float(np.abs(got_s - ref).max())
    checked, flips = rank_flips(pairs, ref, got_b)
    print(f"\n[{label}] {len(pairs)} pairs, {len(docs)} docs, shared_len {sl}, "
          f"cache {kvs[docs[0]].nbytes() / 2**20:.1f} MiB for a {kvs[docs[0]].n_cached}-token prefix")
    print(f"  cached (batched) vs full: max |d log-odds| = {dq_b:.2e}")
    print(f"  cached (single)  vs full: max |d log-odds| = {dq_s:.2e}")
    print(f"  build alone vs padded ({pad_amt} pads): abs {pad_abs:.2e}, rel {pad_rel:.2e}")
    print(f"  shared prefix across docs:          abs {sh_abs:.2e}, rel {sh_rel:.2e} "
          f"(worst: {sh_where[0]}, layer {sh_where[1]}, token {sh_where[2]})")
    print(f"  repeat scoring max diff = {repeat:.2e}; DocKV unchanged = {unchanged}")
    print(f"  rankings: {checked - flips}/{checked} queries identical")

    assert unchanged, "DocKV was modified by scoring"
    assert dq_b <= tol and dq_s <= tol, f"cached vs full exceeds {tol}"
    if strict:
        assert pad_rel <= KV_REL_TOL, f"build padding invariance: rel {pad_rel:.2e} > {KV_REL_TOL}"
        assert sh_rel <= SHARED_REL_TOL, f"shared prefix mismatch: rel {sh_rel:.2e} > {SHARED_REL_TOL}"
        assert repeat == 0.0, "repeat scoring not deterministic on CPU"
        assert flips == 0, "ranking differs"
    print(f"  PASS ({'strict' if strict else 'bf16'} tier)")
    del model


def main():
    cfg = load_config()
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(4)  # leave CPU headroom for the MPS job running alongside
    run(cfg, torch.device("cpu"), "float32", strict=True)
    if "--mps" in sys.argv:
        run(cfg, get_device("mps"), cfg.runtime.dtype, strict=False)


if __name__ == "__main__":
    main()