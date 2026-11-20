# scripts/00_check_env.py
"""Phase 0 environment check. Writes results/env_check.json.

Hard checks (exit code 1 if any fail): arm64 Python, MPS, seeding, Qwen3 yes/no scoring
sanity, padding invariance, cache layout and byte accounting, eager attention weights.
Informational (recorded for decisions): bf16 matmul error, FP8 support, MPS vs CPU drift.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # must be set before transformers is imported

import platform
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marginkv.config import load_config
from marginkv.utils.device import get_device, get_dtype
from marginkv.utils.io import run_info, save_json
from marginkv.utils.seed import set_seed
from marginkv.utils.timing import Timer, summarize_ms

# Qwen3-Reranker prompt template, from the model card.
# Phase 2 moves this into src/marginkv/models/qwen3_reranker.py.
SYSTEM_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

QUERY = "What causes the seasons on Earth?"
DOC_REL = (
    "Earth's seasons are caused by the tilt of its rotational axis relative to its orbital "
    "plane, which changes how directly sunlight reaches each hemisphere during the year."
)
DOC_IRR = (
    "The recipe calls for two cups of flour, a pinch of salt, and one tablespoon of olive oil, "
    "mixed until the dough is smooth and elastic."
)


# ---------- prompt + scoring ----------

def encode_pair(tok, query: str, doc: str) -> list[int]:
    """Tokenize each piece separately, then concatenate ids (never re-tokenize the joined string)."""
    body = f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"
    pre = tok(SYSTEM_PREFIX, add_special_tokens=False)["input_ids"]
    mid = tok(body, add_special_tokens=False)["input_ids"]
    suf = tok(SUFFIX, add_special_tokens=False)["input_ids"]
    return pre + mid + suf


def left_pad(seqs: list[list[int]], pad_id: int):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, T - len(s):] = torch.tensor(s, dtype=torch.long)
        mask[i, T - len(s):] = 1
    # Real tokens start at position 0 in every row, regardless of padding.
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids, mask, pos


@torch.no_grad()
def score_yes(model, seqs, pad_id, yes_id, no_id, device) -> torch.Tensor:
    """P(yes) from the yes/no logits at the last position."""
    ids, mask, pos = left_pad(seqs, pad_id)
    out = model(
        input_ids=ids.to(device),
        attention_mask=mask.to(device),
        position_ids=pos.to(device),
        logits_to_keep=1,
    )
    last = out.logits[:, -1, :].float()
    two = torch.stack([last[:, no_id], last[:, yes_id]], dim=1)
    return torch.log_softmax(two, dim=1)[:, 1].exp().cpu()


# ---------- probes ----------

def _public(obj) -> list[str]:
    return sorted(a for a in dir(obj) if not a.startswith("_"))


@torch.no_grad()
def probe_cache(model, seq: list[int], device) -> dict:
    out = model(input_ids=torch.tensor([seq], device=device), use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    info = {
        "type": type(cache).__name__,
        "module": type(cache).__module__,
        "public_attrs": _public(cache),
    }
    try:
        info["get_seq_length"] = int(cache.get_seq_length())
    except Exception as e:  # recorded, not fatal
        info["get_seq_length"] = f"error: {type(e).__name__}: {e}"

    tensors = []
    layers = getattr(cache, "layers", None)
    if layers is not None and len(layers) > 0:
        l0 = layers[0]
        info["layer_type"] = type(l0).__name__
        info["layer_attrs"] = _public(l0)
        for layer in layers:
            k, v = getattr(layer, "keys", None), getattr(layer, "values", None)
            if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                tensors.append((k, v))
        if tensors:
            info["access"] = "cache.layers[i].keys / cache.layers[i].values"
    if not tensors and hasattr(cache, "key_cache"):
        tensors = list(zip(cache.key_cache, cache.value_cache))
        info["access"] = "cache.key_cache[i] / cache.value_cache[i]"
    if not tensors and isinstance(cache, (tuple, list)):
        tensors = [(k, v) for k, v in cache]
        info["access"] = "legacy tuple"
    if not tensors:
        info["access"] = "UNKNOWN"
        return info

    k0, v0 = tensors[0]
    total = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in tensors)
    info.update(
        num_layers=len(tensors),
        key_shape=list(k0.shape),
        value_shape=list(v0.shape),
        dtype=str(k0.dtype),
        device=str(k0.device),
        seq_len=len(seq),
        bytes_total=total,
        bytes_per_token=total / len(seq),
    )
    return info


@torch.no_grad()
def probe_attention(model, seq: list[int], device) -> dict:
    out = model(input_ids=torch.tensor([seq], device=device), output_attentions=True, logits_to_keep=1)
    att = getattr(out, "attentions", None)
    if not att or att[0] is None:
        return {"ok": False, "reason": "no attentions returned"}
    a0 = att[0].float()
    return {
        "ok": True,
        "num_layers": len(att),
        "shape_layer0": list(a0.shape),  # [batch, q_heads, T, T]
        "rowsum_max_err": float((a0.sum(-1) - 1).abs().max()),
    }


def matmul_check(device) -> dict:
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(1024, 1024, generator=g), torch.randn(1024, 1024, generator=g)
    ref = a @ b  # fp32 CPU reference
    res = {}
    for name, dev, dt in [
        ("cpu_bf16", torch.device("cpu"), torch.bfloat16),
        (f"{device.type}_bf16", device, torch.bfloat16),
        (f"{device.type}_fp32", device, torch.float32),
    ]:
        y = (a.to(dev, dt) @ b.to(dev, dt)).float().cpu()
        res[name] = {
            "max_abs_err": float((y - ref).abs().max()),
            "rel_fro_err": float((y - ref).norm() / ref.norm()),
        }
    return res


def fp8_check(device) -> dict:
    out = {}
    for name in ["float8_e4m3fn", "float8_e5m2"]:
        dt = getattr(torch, name, None)
        if dt is None:
            out[name] = {"ok": False, "error": "dtype missing in this torch"}
            continue
        try:
            x = torch.randn(256, device=device, dtype=torch.bfloat16)
            y = x.to(dt).to(torch.float32)
            out[name] = {"ok": True, "roundtrip_max_abs_err": float((x.float() - y).abs().max())}
        except Exception as e:
            out[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return out


def seed_check(seed: int, device) -> dict:
    res = {}
    for dev in [torch.device("cpu"), device]:
        set_seed(seed)
        a = torch.randn(8, device=dev)
        set_seed(seed)
        b = torch.randn(8, device=dev)
        res[dev.type] = bool(torch.equal(a, b))
    return res


# ---------- main ----------

def main() -> int:
    cfg = load_config()
    set_seed(cfg.seed)
    device = get_device(cfg.runtime.device)
    dtype = get_dtype(cfg.runtime.dtype)
    q = cfg.models.qwen3

    checks: list[tuple[str, bool, str]] = []  # (name, passed, detail)
    report: dict = {"info": run_info(), "device": str(device), "dtype": str(dtype)}

    # Platform
    machine = platform.machine()
    checks.append(("python is arm64", machine == "arm64", machine))
    checks.append(("mps available", torch.backends.mps.is_available(), str(device)))

    seeds = seed_check(cfg.seed, device)
    report["seed"] = seeds
    checks.append(("seeding reproducible", all(seeds.values()), str(seeds)))

    report["matmul"] = matmul_check(device)
    report["fp8"] = {"cpu": fp8_check(torch.device("cpu")), device.type: fp8_check(device)}

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(q.name, revision=q.revision)
    yes_id, no_id = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    pad_id = tok.pad_token_id
    report["tokens"] = {"yes_id": yes_id, "no_id": no_id, "pad_id": pad_id, "unk_id": tok.unk_token_id}
    ids_ok = None not in (yes_id, no_id, pad_id) and tok.unk_token_id not in (yes_id, no_id)
    checks.append(("yes/no/pad ids valid", ids_ok, f"yes={yes_id} no={no_id} pad={pad_id}"))

    seq_rel, seq_irr = encode_pair(tok, QUERY, DOC_REL), encode_pair(tok, QUERY, DOC_IRR)
    report["toy_lengths"] = {"rel": len(seq_rel), "irr": len(seq_irr)}

    # Model on device (bf16, default attention)
    m_dev = AutoModelForCausalLM.from_pretrained(q.name, revision=q.revision, dtype=dtype).to(device).eval()
    report["attn_impl_default"] = getattr(m_dev.config, "_attn_implementation", None)

    dev_batch = score_yes(m_dev, [seq_rel, seq_irr], pad_id, yes_id, no_id, device)
    dev_single = torch.cat([score_yes(m_dev, [s], pad_id, yes_id, no_id, device) for s in (seq_rel, seq_irr)])

    times = []
    for _ in range(7):
        with Timer(device) as t:
            score_yes(m_dev, [seq_rel], pad_id, yes_id, no_id, device)
        times.append(t.ms)
    report["single_pair_latency_ms"] = summarize_ms(times[2:])

    cache_info = probe_cache(m_dev, seq_rel, device)
    report["cache"] = cache_info
    c = m_dev.config
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    expected_bpt = c.num_hidden_layers * c.num_key_value_heads * head_dim * 2 * torch.tensor([], dtype=dtype).element_size()
    report["cache"]["expected_bytes_per_token"] = expected_bpt
    checks.append(("cache layout found", cache_info.get("access") != "UNKNOWN", cache_info.get("access", "")))
    checks.append((
        "cache bytes/token match config",
        cache_info.get("bytes_per_token") == expected_bpt,
        f"measured={cache_info.get('bytes_per_token')} expected={expected_bpt}",
    ))
    del m_dev
    if device.type == "mps":
        torch.mps.empty_cache()

    # CPU fp32 reference with eager attention (also covers the Phase 6 attention check)
    m_cpu = AutoModelForCausalLM.from_pretrained(
        q.name, revision=q.revision, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    cpu_batch = score_yes(m_cpu, [seq_rel, seq_irr], pad_id, yes_id, no_id, "cpu")
    cpu_single = torch.cat([score_yes(m_cpu, [s], pad_id, yes_id, no_id, "cpu") for s in (seq_rel, seq_irr)])
    attn_info = probe_attention(m_cpu, seq_rel, "cpu")
    report["eager_attention"] = attn_info
    del m_cpu

    report["scores"] = {
        f"{device.type}_bf16_batch": dev_batch.tolist(),
        f"{device.type}_bf16_single": dev_single.tolist(),
        "cpu_fp32_batch": cpu_batch.tolist(),
        "cpu_fp32_single": cpu_single.tolist(),
        "pad_invariance_cpu_max_abs": float((cpu_batch - cpu_single).abs().max()),
        f"pad_invariance_{device.type}_max_abs": float((dev_batch - dev_single).abs().max()),
        f"{device.type}_vs_cpu_max_abs": float((dev_batch - cpu_batch).abs().max()),
    }
    s = report["scores"]

    for name, sc in [(f"{device.type} bf16", dev_batch), ("cpu fp32", cpu_batch)]:
        rel, irr = float(sc[0]), float(sc[1])
        checks.append((f"relevant > 0.5 > irrelevant ({name})", rel > 0.5 > irr, f"rel={rel:.4f} irr={irr:.4f}"))
    checks.append((
        "padding invariance (cpu fp32, < 1e-4)",
        s["pad_invariance_cpu_max_abs"] < 1e-4,
        f"{s['pad_invariance_cpu_max_abs']:.2e}",
    ))
    checks.append((
        "eager attention weights (rows sum to 1)",
        bool(attn_info.get("ok")) and attn_info.get("rowsum_max_err", 1.0) < 1e-4,
        f"shape={attn_info.get('shape_layer0')} rowsum_err={attn_info.get('rowsum_max_err')}",
    ))

    report["checks"] = [{"name": n, "passed": p, "detail": d} for n, p, d in checks]
    out = save_json(report, cfg.paths.results_dir / "env_check.json")

    # Summary
    print("\n=== hard checks ===")
    for n, p, d in checks:
        print(f"[{'PASS' if p else 'FAIL'}] {n}: {d}")
    print("\n=== info ===")
    print(f"default attn impl: {report['attn_impl_default']}")
    print(f"cache: {cache_info.get('type')} via {cache_info.get('access')}, "
          f"key_shape={cache_info.get('key_shape')}, dtype={cache_info.get('dtype')}")
    print(f"{device.type} vs cpu score drift: {s[f'{device.type}_vs_cpu_max_abs']:.2e}, "
          f"{device.type} pad drift: {s[f'pad_invariance_{device.type}_max_abs']:.2e}")
    print(f"single pair latency ({device.type}): p50={report['single_pair_latency_ms']['p50']:.1f} ms")
    for where, res in report["fp8"].items():
        for fmt, r in res.items():
            print(f"fp8 {where} {fmt}: " + ("ok, max err %.3f" % r["roundtrip_max_abs_err"] if r["ok"] else r["error"]))
    for k, v in report["matmul"].items():
        print(f"matmul {k}: rel_fro_err={v['rel_fro_err']:.2e}")
    print(f"\nwrote {out}")

    failed = [n for n, p, _ in checks if not p]
    if failed:
        print(f"\n{len(failed)} hard check(s) failed: {failed}")
        return 1
    print("\nall hard checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())