# tests/test_combined.py
"""Phase 7.1 tests: prune-then-quantize transform.

Synthetic (CPU, seconds):
  1. combined == quantize_doc(prune_doc(kv, keep)) exactly, for every compressed format
  2. retention 1.0 == quant only; format bf16 == prune only
  3. shared prefix untouched, next_pos / shared_len unchanged, n_cached = shared + kept
  4. infos pass through from the prune transform
  5. order: int8_tok and KIVI values commute with pruning exactly; KIVI keys do not
  6. packed round trip of combined caches is bit-exact, file size == predicted_file_bytes

Model (CPU fp32, merged doc-first LoRA, 3 SciFact docs; skip with --skip-model):
  7. combined caches in mixed batches == single rows (<= 1e-4 log-odds)
  8. uniform 1.0 + int4_kivi scores == quant-only int4_kivi scores exactly

Run: python tests/test_combined.py [--skip-model]
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import tempfile
from pathlib import Path

import torch

from marginkv.kv import combined as CB
from marginkv.kv import pack
from marginkv.kv import prune as P
from marginkv.kv import quant as Q
from marginkv.kv.cache_ops import DocKV

SHARED = 59
LAYERS = 2
COMPRESSED = [f for f in Q.FORMATS if f != "bf16"]


def synth_kv(T, seed=0, layers=LAYERS, shared=SHARED):
    """bf16 DocKV with real head shapes and an outlier key channel (like post-RoPE keys)."""
    g = torch.Generator().manual_seed(seed)
    n = shared + T
    keys, values = [], []
    for _ in range(layers):
        k = torch.randn(8, n, 128, generator=g)
        k[:, :, 7] *= 40.0
        v = torch.randn(8, n, 128, generator=g) * 0.5
        keys.append(k.to(torch.bfloat16))
        values.append(v.to(torch.bfloat16))
    return DocKV(keys, values, n, shared)


def same(a, b, what=""):
    assert a.next_pos == b.next_pos, (what, a.next_pos, b.next_pos)
    assert a.shared_len == b.shared_len, what
    assert a.num_layers == b.num_layers, what
    for l in range(a.num_layers):
        assert torch.equal(a.keys[l], b.keys[l]), f"{what}: keys differ at layer {l}"
        assert torch.equal(a.values[l], b.values[l]), f"{what}: values differ at layer {l}"


def uniform_keep(T, r):
    keep, info = P.select_keep("uniform", T, r, first=P.FIRST_KEEP, reserve=P.RESERVE)
    return keep, info


def test_compose_matches_manual():
    for T in (241, 70):
        kv = synth_kv(T, seed=T)
        keep, _ = uniform_keep(T, 0.75)
        for fmt in COMPRESSED:
            out = CB.make_combined("uniform", 0.75, fmt)(kv, "d1")
            manual = Q.quantize_doc(P.prune_doc(kv, keep), fmt)
            same(out, manual, f"compose {fmt} T={T}")
            assert out.n_cached == SHARED + len(keep), (fmt, out.n_cached)
            assert out.next_pos == SHARED + T
            assert out.keys[0].dtype == torch.bfloat16


def test_edges():
    T = 241
    kv = synth_kv(T, seed=1)
    for fmt in COMPRESSED:
        r1 = CB.make_combined("uniform", 1.0, fmt)(kv, "d1")
        same(r1, Q.quantize_doc(kv, fmt), f"r=1.0 {fmt} vs quant only")
    pr = P.make_transform("uniform", 0.75)(kv, "d1")
    b = CB.make_combined("uniform", 0.75, "bf16")(kv, "d1")
    same(b, pr, "bf16 vs prune only")
    try:
        CB.make_combined("uniform", 0.75, "int3")
        raise AssertionError("bad format accepted")
    except ValueError:
        pass


def test_shared_prefix_and_input_untouched():
    kv = synth_kv(200, seed=2)
    before = kv.clone()
    for fmt in COMPRESSED:
        out = CB.make_combined("uniform", 0.75, fmt)(kv, "d1")
        for l in range(kv.num_layers):
            assert torch.equal(out.keys[l][:, :SHARED], kv.keys[l][:, :SHARED]), fmt
            assert torch.equal(out.values[l][:, :SHARED], kv.values[l][:, :SHARED]), fmt
    same(kv, before, "input mutated")


def test_infos_passthrough():
    t = CB.make_combined("uniform", 0.75, "int4_kivi")
    assert t.wants_doc is True
    assert t.name == "uniform_r75_int4_kivi", t.name
    assert (t.method, t.retention, t.fmt, t.order) == ("uniform", 0.75, "int4_kivi", "prune_then_quant")
    for i, T in enumerate((241, 70)):
        t(synth_kv(T, seed=10 + i), f"d{i}")
        _, info = uniform_keep(T, 0.75)
        got = t.infos[f"d{i}"]
        assert got["kept"] == info["kept"] and got["target"] == info["target"], (got, info)
    assert set(t.infos) == {"d0", "d1"}


def test_order():
    """quant(prune(kv)) (ours) vs prune(quant(kv)): which formats commute."""
    T = 241
    kv = synth_kv(T, seed=3)
    keep, _ = uniform_keep(T, 0.75)
    res = {}
    for fmt in COMPRESSED:
        ours = Q.quantize_doc(P.prune_doc(kv, keep), fmt)
        other = P.prune_doc(Q.quantize_doc(kv, fmt), keep)
        k_eq = all(torch.equal(ours.keys[l], other.keys[l]) for l in range(LAYERS))
        v_eq = all(torch.equal(ours.values[l], other.values[l]) for l in range(LAYERS))
        res[fmt] = (k_eq, v_eq)
    print("    order commutes (keys, values):", {f: res[f] for f in COMPRESSED})
    assert res["int8_tok"] == (True, True), "int8 per-token should commute exactly"
    assert res["int4_kivi"][1] is True, "KIVI values (per token) should commute"
    assert res["int4_kivi"][0] is False, "KIVI keys (groups over tokens) should not commute"


def test_pack_roundtrip():
    T = 241
    kv = synth_kv(T, seed=4)
    shared_k = [k[:, :SHARED] for k in kv.keys]
    shared_v = [v[:, :SHARED] for v in kv.values]
    with tempfile.TemporaryDirectory() as tmp:
        for fmt in Q.FORMATS:
            out = CB.make_combined("uniform", 0.75, fmt)(kv, "d1")
            path = Path(tmp) / f"{fmt}.safetensors"
            info = pack.pack_doc(out, fmt, path)
            n_kept = out.n_cached - SHARED
            pred = pack.predicted_file_bytes(fmt, n_kept, SHARED, layers=LAYERS, next_pos=out.next_pos)
            assert info["file_bytes"] == pred == path.stat().st_size, (fmt, info["file_bytes"], pred)
            back = pack.unpack_doc(path, shared_k, shared_v)
            same(back, out, f"pack {fmt}")


SYNTHETIC = [
    test_compose_matches_manual,
    test_edges,
    test_shared_prefix_and_input_untouched,
    test_infos_passthrough,
    test_order,
    test_pack_roundtrip,
]


def model_test():
    from marginkv.config import load_config
    from marginkv.kv import cache_ops as C
    from marginkv.kv import sweep as S
    from marginkv.models import qwen3_reranker as qr
    from marginkv.utils.io import load_pt

    torch.set_num_threads(4)
    cfg = load_config()
    dev, dt = torch.device("cpu"), torch.float32
    tok_dir = cfg.paths.artifacts_dir / "tokenized"
    art = load_pt(tok_dir / "scifact_qwen3.pt")
    dfart = load_pt(tok_dir / "scifact_qwen3_df.pt")
    assert dfart["base_content_sha256"] == art["content_sha256"]
    pieces, docs, by_doc = S.sweep_plan(art, dfart)
    s = C.shared_prefix_len(pieces)
    assert s == SHARED, s
    model = qr.load_model(str(cfg.paths.artifacts_dir / "lora" / "df" / "merged"), None, dev, dt)

    picked = [d for d in docs if len(by_doc[d]) >= 2][:3]
    base, news = {}, {}
    for d in picked:
        seqs = []
        for qid in by_doc[d][:2]:
            prefix, new = C.split_doc_first(pieces, dfart["doc_seam"][d], art["query_df"][qid], variant="B")
            seqs.append(new)
        base[d] = C.build_doc_kvs(model, [prefix], dev, s)[0]
        news[d] = seqs

    arms = {
        "full": lambda kv, d: kv,
        "r75": P.make_transform("uniform", 0.75),
        "kivi": lambda kv, d: Q.quantize_doc(kv, "int4_kivi"),
        "r75_fp8": CB.make_combined("uniform", 0.75, "fp8_tensor"),
        "r75_kivi": CB.make_combined("uniform", 0.75, "int4_kivi"),
        "r100_kivi": CB.make_combined("uniform", 1.0, "int4_kivi"),
    }
    rows = []  # (arm, doc, j, cache, new_ids); doc-major inside arm-interleaved order so batches mix
    for d in picked:
        caches = {a: f(base[d], d) for a, f in arms.items()}
        for j, new in enumerate(news[d]):
            for a in arms:
                rows.append((a, d, j, caches[a], new))
    print(f"    {len(picked)} docs, {len(rows)} rows, cache lengths "
          f"{sorted({r[3].n_cached for r in rows})}")

    batched = []
    for i in range(0, len(rows), 8):
        chunk = rows[i:i + 8]
        out = C.score_on_caches(model, [r[3] for r in chunk], [r[4] for r in chunk], dev)
        batched += out["log_odds"]
    single = [C.score_on_caches(model, [r[3]], [r[4]], dev)["log_odds"][0] for r in rows]
    diff = max(abs(a - b) for a, b in zip(batched, single))
    print(f"    mixed batch vs single: max {diff:.2e}")
    assert diff <= 1e-4, diff

    sc = {(r[0], r[1], r[2]): x for r, x in zip(rows, single)}
    keys = [(d, j) for d in picked for j in range(2)]
    exact = max(abs(sc[("r100_kivi", d, j)] - sc[("kivi", d, j)]) for d, j in keys)
    print(f"    r100_kivi vs kivi: max {exact:.1e}")
    assert exact == 0.0, exact
    for a in ("r75", "kivi", "r75_fp8", "r75_kivi"):
        m = max(abs(sc[(a, d, j)] - sc[("full", d, j)]) for d, j in keys)
        print(f"    shift vs full, {a:9s}: max {m:.3f} log-odds")
        assert all(torch.isfinite(torch.tensor(sc[(a, d, j)])) for d, j in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()
    for t in SYNTHETIC:
        t()
        print(f"ok  {t.__name__}")
    if not args.skip_model:
        model_test()
        print("ok  model_test")
    print("all passed")


if __name__ == "__main__":
    main()