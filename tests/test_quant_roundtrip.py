# tests/test_quant_roundtrip.py
"""Phase 5.3 quantizer tests.

Default (CPU, synthetic KV with key outlier channels, seconds):
  structure (shapes, dtype, next_pos, shared_len), shared prefix bit-identical, input not mutated,
  deterministic, per-element error bounds for every encoder (fp32 decode), partial key groups,
  constant groups (no NaN, exact), error order int8 < fp8 < int4, KIVI keys beat per-token keys on
  outlier channels, theory_bytes ratios.
--mps: one real SciFact doc KV from the merged doc-first LoRA (MPS bf16). Every format on MPS vs on
  a CPU copy (agreement), all finite, per-layer relative error table.
  Writes results/metrics/quant_check.json.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys

import torch

from marginkv.kv import quant as Q
from marginkv.kv.cache_ops import DocKV

SHARED = 59


def synth_kv(n_doc, L=3, H=8, D=128, seed=0, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    n = SHARED + n_doc
    keys, values = [], []
    for _ in range(L):
        k = torch.randn(H, n, D, generator=g) * 2
        k[:, :, :4] *= 40                      # outlier channels, same in every token
        keys.append(k.to(dtype))
        values.append(torch.randn(H, n, D, generator=g).to(dtype))
    return DocKV(keys, values, next_pos=n, shared_len=SHARED)


def rel(stats, role):
    return (sum(stats[f"{role}_err2"]) / sum(stats[f"{role}_ref2"])) ** 0.5


def test_structure_and_shared():
    for n_doc in (257, 40, 5, 0):
        kv = synth_kv(n_doc)
        snap = kv.clone()
        for fmt in Q.FORMATS:
            q = Q.quantize_doc(kv, fmt)
            if fmt == "bf16":
                assert q is kv
                continue
            assert q.num_layers == kv.num_layers and q.next_pos == kv.next_pos and q.shared_len == kv.shared_len
            for a, b in zip(kv.keys + kv.values, q.keys + q.values):
                assert b.shape == a.shape and b.dtype == a.dtype, (fmt, n_doc)
                assert torch.equal(a[:, :SHARED], b[:, :SHARED]), f"shared prefix changed ({fmt})"
                assert torch.isfinite(b.float()).all(), f"non-finite ({fmt}, {n_doc})"
            q2 = Q.quantize_doc(kv, fmt)
            assert all(torch.equal(a, b) for a, b in zip(q.keys + q.values, q2.keys + q2.values)), "not deterministic"
        assert all(torch.equal(a, b) for a, b in zip(kv.keys + kv.values, snap.keys + snap.values)), "input mutated"
    print("structure, shared prefix, determinism, no mutation: PASS")


def test_bounds():
    x = synth_kv(257, L=1).keys[0][:, SHARED:].float()
    for per_head in (False, True):
        e = Q.fp8_encode(x, per_head)
        s = e["scale"].float()
        err = (Q.fp8_decode(e, torch.float32) - x).abs()
        assert (err <= 2 ** -4 * x.abs() + 0.25 * s + 1e-6).all(), f"fp8 bound (per_head={per_head})"
    e = Q.int_sym_encode(x, 8)
    err = (Q.int_sym_decode(e, torch.float32) - x).abs()
    assert (err <= 0.6 * e["scale"].float() + 1e-6).all(), "int8 bound"
    for dim in (Q.TOKEN_DIM, Q.CHANNEL_DIM):
        e = Q.int_asym_group_encode(x, 4, 32, dim)
        sizes = Q.group_sizes(x.shape[dim], 32)
        s = Q.expand_groups(e["scale"].float(), sizes, dim)
        mn = Q.expand_groups(e["min"].float(), sizes, dim)
        err = (Q.int_asym_group_decode(e, 32, dim, torch.float32) - x).abs()
        assert (err <= 0.5 * s + 15 * s * 2 ** -10 + 2 ** -10 * mn.abs() + 1e-6).all(), f"int4 bound (dim={dim})"
    print("per-element error bounds (fp8 x2, int8, int4 token/channel groups): PASS")


def test_groups_and_constant():
    for n_doc, groups in ((257, 9), (256, 8), (40, 2), (5, 1)):
        x = synth_kv(n_doc, L=1).keys[0][:, SHARED:]
        e = Q.int_asym_group_encode(x, 4, 32, Q.TOKEN_DIM)
        assert e["scale"].shape == (8, groups, 128) and e["min"].shape == (8, groups, 128), (n_doc, e["scale"].shape)
        assert e["codes"].shape == x.shape and int(e["codes"].max()) <= 15
    c = torch.full((8, 64, 128), 3.25, dtype=torch.bfloat16)
    for fmt in Q.FORMATS[1:]:
        for role in ("k", "v"):
            y = Q.decode(Q.encode(c, fmt, role), fmt, role, torch.bfloat16)
            assert torch.isfinite(y.float()).all() and torch.equal(y, c), f"constant input not exact ({fmt}, {role})"
    z = torch.zeros(8, 64, 128, dtype=torch.bfloat16)
    for fmt in Q.FORMATS[1:]:
        y = Q.decode(Q.encode(z, fmt, "k"), fmt, "k", torch.bfloat16)
        assert torch.equal(y, z), f"zeros not exact ({fmt})"
    print("partial key groups (257 -> 9 groups), constant and zero inputs exact: PASS")


def test_error_order_and_kivi():
    kv = synth_kv(257)
    r = {fmt: Q.recon_stats(kv, Q.quantize_doc(kv, fmt)) for fmt in Q.FORMATS[1:]}
    v = {fmt: rel(r[fmt], "v") for fmt in r}
    k = {fmt: rel(r[fmt], "k") for fmt in r}
    assert v["int8_tok"] < v["fp8_head"] < v["int4_kivi"], f"value error order off: {v}"
    assert k["int4_kivi"] < k["int4_tok"], f"KIVI keys should beat per-token keys on outlier channels: {k}"
    print("rel error, values: " + ", ".join(f"{f} {x:.4f}" for f, x in v.items()))
    print("rel error, keys:   " + ", ".join(f"{f} {x:.4f}" for f, x in k.items()))
    print("error order int8 < fp8 < int4, KIVI keys < per-token keys: PASS")


def test_theory_bytes():
    bf = Q.theory_bytes("bf16", 256)
    ratios = {f: bf / Q.theory_bytes(f, 256) for f in Q.FORMATS}
    assert abs(ratios["int4_kivi"] - 3.2) < 1e-9 and abs(ratios["int4_tok"] - 3.2) < 1e-9, ratios
    assert abs(ratios["int8_tok"] - 16 / 8.125) < 1e-9, ratios
    assert 1.999 < ratios["fp8_tensor"] <= 2.0 and 1.999 < ratios["fp8_head"] <= 2.0, ratios
    r257 = {f: bf / Q.theory_bytes(f, 256) * Q.theory_bytes("bf16", 257) / bf
            for f in Q.FORMATS}
    r257 = {f: Q.theory_bytes("bf16", 257) / Q.theory_bytes(f, 257) for f in Q.FORMATS}
    print("theory ratio vs bf16 (n_doc 257): " + ", ".join(f"{f} {x:.3f}x" for f, x in r257.items()))
    print("theory_bytes: PASS")


def real_check():
    from marginkv.config import load_config
    from marginkv.kv import cache_ops as co
    from marginkv.models import qwen3_reranker as qr
    from marginkv.train.data import sha256_file
    from marginkv.utils.io import load_json, load_pt, run_info, save_json

    cfg = load_config()
    torch.set_grad_enabled(False)
    A = cfg.paths.artifacts_dir
    model_dir = A / "lora" / "df" / "merged"
    shas = load_json(model_dir / "marginkv_merge.json")["files_sha256"]
    for fname, want in shas.items():
        assert sha256_file(model_dir / fname) == want, f"sha mismatch: {fname}"
    device = torch.device("mps")
    model = qr.load_model(str(model_dir), None, device, torch.bfloat16)
    art = load_pt(A / "tokenized" / "scifact_qwen3.pt")
    dfart = load_pt(A / "tokenized" / "scifact_qwen3_df.pt")
    pieces = {**art["pieces"], **dfart["seam_pieces"]}
    doc = next(d for d in art["doc_order"] if dfart["cache_len_seam"][d] == max(dfart["cache_len_seam"].values()))
    prefix, _ = co.split_doc_first(pieces, dfart["doc_seam"][doc], [], "B")
    kv = co.build_doc_kvs(model, [prefix], device, co.shared_prefix_len(pieces))[0]
    kv_cpu = kv.to("cpu")

    out = {"run_info": run_info(), "phase": "5.3", "doc_id": doc, "n_cached": kv.n_cached,
           "n_doc": kv.n_cached - kv.shared_len, "formats": {}}
    for fmt in Q.FORMATS[1:]:
        qm, qc = Q.quantize_doc(kv, fmt), Q.quantize_doc(kv_cpu, fmt)
        mism, tot, maxd = 0, 0, 0.0
        for a, b in zip(qm.keys + qm.values, qc.keys + qc.values):
            a = a.cpu()
            assert torch.isfinite(a.float()).all(), f"non-finite on MPS ({fmt})"
            ne = a != b
            mism += int(ne.sum())
            tot += a.numel()
            maxd = max(maxd, float((a.float() - b.float()).abs().max()))
        st = Q.recon_stats(kv, qm)
        k_rel = [(e / r) ** 0.5 for e, r in zip(st["k_err2"], st["k_ref2"])]
        v_rel = [(e / r) ** 0.5 for e, r in zip(st["v_err2"], st["v_ref2"])]
        out["formats"][fmt] = {"mps_cpu_mismatch_frac": mism / tot, "mps_cpu_max_abs": maxd,
                               "k_rel": rel(st, "k"), "v_rel": rel(st, "v"),
                               "k_rel_per_layer": k_rel, "v_rel_per_layer": v_rel,
                               "k_rel_worst_layer": int(max(range(len(k_rel)), key=k_rel.__getitem__)),
                               "v_rel_worst_layer": int(max(range(len(v_rel)), key=v_rel.__getitem__))}
        f = out["formats"][fmt]
        print(f"[real {fmt:10s}] K rel {f['k_rel']:.4f} (worst L{f['k_rel_worst_layer']} {max(k_rel):.4f})  "
              f"V rel {f['v_rel']:.4f} (worst L{f['v_rel_worst_layer']} {max(v_rel):.4f})  "
              f"MPS vs CPU mismatch {f['mps_cpu_mismatch_frac']:.1e} max {maxd:.2e}")
        assert f["mps_cpu_mismatch_frac"] <= 1e-3, f"MPS and CPU disagree too often ({fmt})"
    kmax = max(float(t.float().abs().max()) for t in kv.keys)
    vmax = max(float(t.float().abs().max()) for t in kv.values)
    out["kv_absmax"] = {"k": kmax, "v": vmax}
    print(f"real KV |max|: keys {kmax:.1f}, values {vmax:.1f} (fp16 range 65504)")
    save_json(out, cfg.paths.results_dir / "metrics" / "quant_check.json")
    print("real KV check: PASS")


if __name__ == "__main__":
    test_structure_and_shared()
    test_bounds()
    test_groups_and_constant()
    test_error_order_and_kivi()
    test_theory_bytes()
    if "--mps" in sys.argv:
        real_check()