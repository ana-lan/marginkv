# tests/test_prune.py
"""Phase 6.1: pruning selection, importance math, and exactness of pruned-cache scoring.

Synthetic checks always run. Model checks (CPU fp32, merged doc-first LoRA):
  pruned cache == full prompt with a 4D mask hiding pruned doc tokens from the
  query/suffix rows (<= 1e-4 log-odds), keep-all == unpruned exactly, mixed-length
  batches == single rows. Run: python tests/test_prune.py [--skip-model]
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import time

import torch

from marginkv.config import load_config
from marginkv.utils.io import load_pt
from marginkv.utils.seed import set_seed
from marginkv.kv import prune as P
from marginkv.kv.cache_ops import DocKV, build_doc_kvs, score_on_caches, shared_prefix_len, split_doc_first
from marginkv.models import qwen3_reranker as qr

TOL = 1e-4


# ---------------- synthetic ----------------

def test_budget_and_mandatory():
    assert P.round_half_up(2.5) == 3 and P.round_half_up(2.49) == 2
    assert P.mandatory_idx(5) == [0, 1, 2, 3, 4]
    assert P.mandatory_idx(100) == [0] + list(range(92, 100))
    assert P.budget(100, 0.25, 9) == (25, 25)
    assert P.budget(21, 0.25, 9) == (5, 9)  # forced
    keep, info = P.select_keep("uniform", 21, 0.25)
    assert info["forced"] and info["kept"] == 9 and keep.tolist() == P.mandatory_idx(21)
    keep, info = P.select_keep("random", 5, 0.5, seed=1)
    assert keep.tolist() == [0, 1, 2, 3, 4]


def test_methods_basic():
    T = 257
    imp = torch.rand(T, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    imp[-1] = float("nan")  # last token undefined, covered by the reserve
    mand = set(P.mandatory_idx(T))
    for m in P.METHODS:
        for r in P.RETENTIONS:
            kw = {"importance": imp} if m.startswith("attn") else {"seed": 7} if m == "random" else {}
            keep, info = P.select_keep(m, T, r, **kw)
            kl = keep.tolist()
            assert kl == sorted(set(kl)), m
            assert mand <= set(kl), m
            assert len(kl) == P.round_half_up(r * T) == info["kept"], (m, r)
            keep2, _ = P.select_keep(m, T, r, **kw)
            assert torch.equal(keep, keep2), "not deterministic"
    a, _ = P.select_keep("random", T, 0.5, seed=P.doc_seed(42, "d1"))
    b, _ = P.select_keep("random", T, 0.5, seed=P.doc_seed(42, "d2"))
    assert not torch.equal(a, b)
    assert P.doc_seed(42, "d1") == P.doc_seed(42, "d1")


def test_uniform_and_boundary_shape():
    keep, _ = P.select_keep("uniform", 100, 0.5)
    free = sorted(set(keep.tolist()) - set(P.mandatory_idx(100)))
    gaps = [b - a for a, b in zip(free, free[1:])]
    assert max(gaps) - min(gaps) <= 1, gaps  # evenly spaced
    keep, _ = P.select_keep("boundary", 100, 0.25)
    kl = keep.tolist()
    assert kl[:4] == [0, 1, 2, 3]
    tail = kl[4:]
    assert tail == list(range(100 - len(tail), 100)), "boundary tail not contiguous"


def test_attn_ranking_ties():
    T = 40  # budget 20, mandatory 9 (0 and 32..39), so 11 free slots
    imp = torch.zeros(T, dtype=torch.float64)
    imp[5] = 2.0
    imp[9] = 1.0
    keep, _ = P.select_keep("attn_a", T, 0.5, importance=imp)
    kl = keep.tolist()
    assert 5 in kl and 9 in kl
    # remaining free slots are ties (0.0) -> lowest indices first
    mand = P.mandatory_idx(T)
    rest = [i for i in kl if i not in mand and i not in (5, 9)]
    expected = [i for i in range(T) if i not in mand and i not in (5, 9)][:len(rest)]
    assert rest == expected, (rest, expected)
    assert rest == [1, 2, 3, 4, 6, 7, 8, 10, 11]


def test_importance_hand():
    s = 1
    A = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],   # doc j=0
        [0.3, 0.2, 0.5, 0.0],   # doc j=1
        [0.2, 0.3, 0.1, 0.4],   # doc j=2
    ], dtype=torch.float64)
    a = P.importance_from_attn(A, s, weighted=False)
    assert abs(a[0] - 0.25) < 1e-12 and abs(a[1] - 0.1) < 1e-12 and torch.isnan(a[2])
    b = P.importance_from_attn(A, s, weighted=True)  # weights = abs pos + 1: j=1 -> 3, j=2 -> 4
    assert abs(b[0] - 0.9) < 1e-12 and abs(b[1] - 0.4) < 1e-12 and torch.isnan(b[2])


def test_importance_uniform_b_is_one():
    s, T = 59, 40
    S = s + T
    A = torch.tril(torch.ones(S, S, dtype=torch.float64))
    A = A / A.sum(1, keepdim=True)  # row p attends 1/(p+1) to every key
    b = P.importance_from_attn(A, s, weighted=True)
    assert torch.allclose(b[:-1], torch.ones(T - 1, dtype=torch.float64)), b[:5]
    a = P.importance_from_attn(A, s, weighted=False)
    assert torch.all(a[:-2] > a[1:-1]), "(a) should fall toward the end under uniform attention (early-token bias)"


def test_h2o_and_transform():
    s = 1
    A = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.3, 0.2, 0.5, 0.0],
        [0.2, 0.3, 0.1, 0.4],
    ], dtype=torch.float64)
    a = P.importance_from_attn(A, s, weighted=False)
    h = P.importance_h2o(a)
    assert abs(h[0] - 0.5) < 1e-12 and abs(h[1] - 0.1) < 1e-12 and torch.isnan(h[2])  # plain column sums

    s, T = 3, 30
    kv = DocKV(keys=[torch.randn(8, s + T, 128)], values=[torch.randn(8, s + T, 128)], next_pos=s + T, shared_len=s)
    imp = {"d": torch.arange(T, dtype=torch.float64)}
    for m in ("uniform", "boundary", "attn_a", "attn_h2o", "random"):
        fn = P.make_transform(m, 0.5, importance=imp if m.startswith("attn") else None, seed=42)
        pk = fn(kv, "d")
        assert fn.wants_doc and fn.infos["d"]["kept"] == P.kept_doc_tokens(pk) == 15, m
    keep_all = P.make_transform("uniform", 1.0)
    pk = keep_all(kv, "d")
    assert all(torch.equal(x, y) for x, y in zip(pk.keys, kv.keys)) and pk.n_cached == kv.n_cached


def test_pack_pruned():
    import pathlib
    import tempfile
    from marginkv.kv import pack as K
    from marginkv.kv import quant as Q
    s, T, L = 5, 70, 2
    g = torch.Generator().manual_seed(0)
    keys = [torch.randn(8, s + T, 128, generator=g).to(torch.bfloat16) for _ in range(L)]
    vals = [torch.randn(8, s + T, 128, generator=g).to(torch.bfloat16) for _ in range(L)]
    kv = DocKV(keys=keys, values=vals, next_pos=s + T, shared_len=s)
    keep, _ = P.select_keep("random", T, 0.5, seed=3)
    pk = P.prune_doc(kv, keep)
    sk, sv = [k[:, :s] for k in kv.keys], [v[:, :s] for v in kv.values]
    with tempfile.TemporaryDirectory() as tmp:
        for fmt in ("bf16", "fp8_tensor", "int8_tok", "int4_kivi"):
            path = pathlib.Path(tmp) / f"{fmt}.safetensors"
            ref = pk if fmt == "bf16" else Q.quantize_doc(pk, fmt)
            info = K.pack_doc(pk, fmt, path)
            back = K.unpack_doc(path, sk, sv)
            assert back.next_pos == s + T and back.shared_len == s and back.n_cached == pk.n_cached, fmt
            assert all(torch.equal(a, b) for a, b in zip(back.keys, ref.keys)), f"{fmt} keys not bit-exact"
            assert all(torch.equal(a, b) for a, b in zip(back.values, ref.values)), f"{fmt} values not bit-exact"
            want = K.predicted_file_bytes(fmt, len(keep), s, L, 8, 128, "bfloat16", next_pos=s + T)
            assert info["file_bytes"] == want, (fmt, info["file_bytes"], want)

    
def test_prune_doc_synthetic():
    s, T, L = 3, 10, 2
    keys = [torch.randn(8, s + T, 128) for _ in range(L)]
    vals = [torch.randn(8, s + T, 128) for _ in range(L)]
    kv = DocKV(keys=keys, values=vals, next_pos=s + T, shared_len=s)
    before = [k.clone() for k in kv.keys]
    keep = torch.tensor([0, 4, 7, 9])
    pk = P.prune_doc(kv, keep)
    assert pk.n_cached == s + 4 and pk.next_pos == s + T and pk.shared_len == s
    assert P.doc_len(pk) == T and P.kept_doc_tokens(pk) == 4
    for l in range(L):
        assert torch.equal(pk.keys[l][:, :s], kv.keys[l][:, :s])
        assert torch.equal(pk.keys[l][:, s:], kv.keys[l][:, s + keep])
        assert torch.equal(pk.values[l][:, s:], kv.values[l][:, s + keep])
        assert torch.equal(kv.keys[l], before[l]), "input mutated"
    full = P.prune_doc(kv, torch.arange(T))
    assert all(torch.equal(a, b) for a, b in zip(full.keys, kv.keys))
    for bad in ([3, 1], [0, 0], [10], []):
        try:
            P.prune_doc(kv, torch.tensor(bad, dtype=torch.long))
            raise AssertionError(f"accepted bad keep {bad}")
        except ValueError:
            pass
    try:
        P.prune_doc(pk, torch.tensor([0]))
        raise AssertionError("accepted an already pruned cache")
    except ValueError:
        pass


# ---------------- model (CPU fp32) ----------------

@torch.no_grad()
def masked_reference(model, ids, next_pos, shared_len, pruned_doc):
    """Full prompt forward; query/suffix rows can't see pruned doc tokens. Returns log-odds."""
    L = len(ids)
    allowed = torch.tril(torch.ones(L, L, dtype=torch.bool))
    if len(pruned_doc):
        cols = torch.as_tensor(pruned_doc, dtype=torch.long) + shared_len
        allowed[next_pos:, cols] = False
    mask = torch.where(allowed, 0.0, torch.finfo(torch.float32).min)[None, None]
    x = torch.tensor([ids])
    pos = torch.arange(L)[None]
    h = model.model(input_ids=x, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state[:, -1]
    W = model.lm_head.weight[[qr.NO_ID, qr.YES_ID]].float()
    z = h.float() @ W.T
    return (z[:, 1] - z[:, 0]).item()


def lo(model, kvs, news, device):
    return [float(x) for x in score_on_caches(model, kvs, news, device)["log_odds"]]


def test_model():
    cfg = load_config()
    set_seed(cfg.seed)
    torch.set_num_threads(4)
    ds = cfg.datasets[0]
    art = load_pt(cfg.paths.artifacts_dir / "tokenized" / f"{ds}_qwen3.pt")
    dfart = load_pt(cfg.paths.artifacts_dir / "tokenized" / f"{ds}_qwen3_df.pt")
    assert dfart["base_content_sha256"] == art["content_sha256"]
    pieces = {**art["pieces"], **dfart["seam_pieces"]}
    s = shared_prefix_len(pieces)

    device, dtype = torch.device("cpu"), torch.float32
    t0 = time.time()
    model = qr.load_model(str(cfg.paths.artifacts_dir / "lora" / "df" / "merged"), None, device, dtype)
    print(f"[model] loaded in {time.time() - t0:.1f}s, shared_len {s}")

    qids = art["query_order"][:2]
    docs = art["rerank_candidates"][qids[0]][:3]
    kvs, prefixes, news = {}, {}, {}
    for d in docs:
        for q in qids:
            pre, new = split_doc_first(pieces, dfart["doc_seam"][d], art["query_df"][q], variant="B")
            prefixes[d] = pre
            news[(d, q)] = new
        kvs[d] = build_doc_kvs(model, [prefixes[d]], device, s)[0]
        kv = kvs[d]
        assert kv.n_cached == kv.next_pos == len(prefixes[d])
        assert P.doc_len(kv) == len(dfart["doc_seam"][d])
        assert tuple(kv.keys[0].shape) == (8, kv.n_cached, 128)

    worst_ref, worst_batch, shifts = 0.0, 0.0, []
    batch_kvs, batch_news, singles = [], [], []
    for di, d in enumerate(docs):
        kv, T = kvs[d], P.doc_len(kvs[d])
        snapshot = [k.clone() for k in kv.keys]
        q = qids[0]
        ids = prefixes[d] + news[(d, q)]
        base = lo(model, [kv], [news[(d, q)]], device)[0]

        # 0. unpruned cache vs 4D-mask reference with nothing hidden (validates the 4D path)
        ref0 = masked_reference(model, ids, kv.next_pos, s, [])
        worst_ref = max(worst_ref, abs(base - ref0))

        # 1. keep-all is exact
        full = P.prune_doc(kv, torch.arange(T))
        assert lo(model, [full], [news[(d, q)]], device)[0] == base, "keep-all changed the score"

        # 2. pruned cache == masked full prompt
        imp = torch.rand(T, generator=torch.Generator().manual_seed(di), dtype=torch.float64)
        imp[-1] = float("nan")
        for m in ("uniform", "boundary", "random", "attn_a"):
            for r in (0.25, 0.5):
                kw = {"importance": imp} if m == "attn_a" else {"seed": P.doc_seed(cfg.seed, d)} if m == "random" else {}
                keep, info = P.select_keep(m, T, r, **kw)
                pk = P.prune_doc(kv, keep)
                got = lo(model, [pk], [news[(d, q)]], device)[0]
                pruned = sorted(set(range(T)) - set(keep.tolist()))
                ref = masked_reference(model, ids, kv.next_pos, s, pruned)
                err = abs(got - ref)
                worst_ref = max(worst_ref, err)
                shifts.append(abs(got - base))
                print(f"  {d} T={T} {m:8s} r={r:.2f} kept={info['kept']:3d} cached={got:+.5f} ref={ref:+.5f} err={err:.1e} shift={got - base:+.3f}")
                assert err <= TOL, f"pruned cache != masked reference ({err:.2e})"
                if r == 0.25:
                    for q2 in qids:
                        batch_kvs.append(pk)
                        batch_news.append(news[(d, q2)])
        batch_kvs.append(kv)  # an unpruned row in the same batch
        batch_news.append(news[(d, qids[1])])
        assert all(torch.equal(a, b) for a, b in zip(kv.keys, snapshot)), "DocKV mutated by pruning/scoring"

    # 3. mixed-length batch == single rows
    singles = [lo(model, [k], [n], device)[0] for k, n in zip(batch_kvs, batch_news)]
    for start in range(0, len(batch_kvs), 8):
        got = lo(model, batch_kvs[start:start + 8], batch_news[start:start + 8], device)
        for g, sgl in zip(got, singles[start:start + 8]):
            worst_batch = max(worst_batch, abs(g - sgl))
    lens = sorted({k.n_cached for k in batch_kvs})
    print(f"[model] max |cached - masked ref| {worst_ref:.2e}; mixed batch vs single {worst_batch:.2e} "
          f"over {len(batch_kvs)} rows, cache lens {lens[0]}..{lens[-1]}; max |shift| {max(shifts):.3f}")
    assert worst_batch <= TOL, f"mixed batch differs ({worst_batch:.2e})"
    assert max(shifts) > 1e-3, "pruning never changed a score: is the mask or the cache being ignored?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()
    for f in (test_budget_and_mandatory, test_methods_basic, test_uniform_and_boundary_shape,
              test_attn_ranking_ties, test_importance_hand, test_importance_uniform_b_is_one,
              test_h2o_and_transform, test_pack_pruned,
              test_prune_doc_synthetic):
        f()
        print(f"PASS {f.__name__}")
    if not args.skip_model:
        test_model()
        print("PASS test_model")
    print("ALL PASS")


if __name__ == "__main__":
    main()