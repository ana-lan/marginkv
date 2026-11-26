# src/marginkv/kv/quant.py
"""KV cache quantization for stored doc KV (Phase 5.3).

Every format = encode (codes + fp16 scales/mins) then decode back to the model dtype. Fake quant
(quantize_doc) and packed storage (Phase 5.5) call the same encode/decode, so unpacking a file
reproduces the fake-quant tensors bit for bit. Scales and mins are rounded to fp16 *before* the codes
are computed, because fp16 is what gets stored.

Only doc tokens [shared_len:] are quantized. The shared system + instruction prefix stays in the model
dtype: it is stored once for the whole corpus and holds the attention-sink tokens.

Per-layer tensors are [H, n, D] (8 KV heads, tokens, head_dim 128). Formats:
  bf16        no-op (returns the input DocKV)
  fp8_tensor  e4m3fn, one scale per layer and role (amax / 448)
  fp8_head    e4m3fn, one scale per layer, role and KV head
  int8_tok    symmetric int8, one scale per (head, token) over head_dim (amax / 127)
  int4_kivi   asymmetric int4 (scale + min), group 32. Keys: per (head, channel), groups of 32 tokens
              (last group may be shorter). Values: per (head, token), groups of 32 channels. (KIVI)
  int4_tok    asymmetric int4, group 32 along head_dim for keys and values (diagnostic: what
              per-channel keys buy)
Keys are quantized after RoPE (that is what is stored).
"""
from __future__ import annotations

import math

import torch

from marginkv.kv.cache_ops import DocKV

FORMATS = ("bf16", "fp8_tensor", "fp8_head", "int8_tok", "int4_kivi", "int4_tok")
FP8_MAX = 448.0
GROUP = 32
TOKEN_DIM, CHANNEL_DIM = 1, 2   # axes of a per-layer [H, n, D] tensor

SPEC = {
    "fp8_tensor": {"k": ("fp8", {"per_head": False}), "v": ("fp8", {"per_head": False})},
    "fp8_head": {"k": ("fp8", {"per_head": True}), "v": ("fp8", {"per_head": True})},
    "int8_tok": {"k": ("sym", {"bits": 8}), "v": ("sym", {"bits": 8})},
    "int4_kivi": {"k": ("asym", {"bits": 4, "group": GROUP, "dim": TOKEN_DIM}),
                  "v": ("asym", {"bits": 4, "group": GROUP, "dim": CHANNEL_DIM})},
    "int4_tok": {"k": ("asym", {"bits": 4, "group": GROUP, "dim": CHANNEL_DIM}),
                 "v": ("asym", {"bits": 4, "group": GROUP, "dim": CHANNEL_DIM})},
}


def _f16_scale(s: torch.Tensor) -> torch.Tensor:
    """Round positive scales to fp16; zero (constant data) or fp16 underflow -> 1."""
    s16 = s.to(torch.float16)
    return torch.where(s16 > 0, s16, torch.ones_like(s16))


def group_sizes(n: int, group: int) -> list[int]:
    return [group] * (n // group) + ([n % group] if n % group else [])


def expand_groups(t: torch.Tensor, sizes: list[int], dim: int) -> torch.Tensor:
    """t has one slice per group along dim; repeat slice i to sizes[i] (expand + cat, MPS-safe)."""
    shape = list(t.shape)
    parts = []
    for i, size in enumerate(sizes):
        shape[dim] = size
        parts.append(t.narrow(dim, i, 1).expand(shape))
    return torch.cat(parts, dim=dim)


# ---- FP8 e4m3fn ----

def fp8_encode(x: torch.Tensor, per_head: bool) -> dict:
    xf = x.float()
    amax = xf.abs().amax(dim=(1, 2), keepdim=True) if per_head else xf.abs().amax().reshape(1, 1, 1)
    scale = _f16_scale(amax / FP8_MAX)
    codes = (xf / scale.float()).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return {"codes": codes, "scale": scale}


def fp8_decode(enc: dict, dtype) -> torch.Tensor:
    return (enc["codes"].float() * enc["scale"].float()).to(dtype)


# ---- symmetric int, per (head, token) ----

def int_sym_encode(x: torch.Tensor, bits: int = 8) -> dict:
    qmax = 2 ** (bits - 1) - 1
    xf = x.float()
    scale = _f16_scale(xf.abs().amax(dim=CHANNEL_DIM, keepdim=True) / qmax)   # [H, n, 1]
    q = torch.round(xf / scale.float()).clamp(-qmax, qmax).to(torch.int8)
    return {"codes": q, "scale": scale}


def int_sym_decode(enc: dict, dtype) -> torch.Tensor:
    return (enc["codes"].float() * enc["scale"].float()).to(dtype)


# ---- asymmetric int, groups along one axis ----

def int_asym_group_encode(x: torch.Tensor, bits: int, group: int, dim: int) -> dict:
    qmax = 2 ** bits - 1
    xf = x.float()
    sizes = group_sizes(xf.shape[dim], group)
    chunks = torch.split(xf, sizes, dim=dim)
    mn = torch.cat([c.amin(dim=dim, keepdim=True) for c in chunks], dim=dim)
    mx = torch.cat([c.amax(dim=dim, keepdim=True) for c in chunks], dim=dim)
    scale = _f16_scale((mx - mn) / qmax)
    mn16 = mn.to(torch.float16)
    q = torch.round((xf - expand_groups(mn16.float(), sizes, dim)) / expand_groups(scale.float(), sizes, dim))
    return {"codes": q.clamp(0, qmax).to(torch.uint8), "scale": scale, "min": mn16}


def int_asym_group_decode(enc: dict, group: int, dim: int, dtype) -> torch.Tensor:
    q = enc["codes"]
    sizes = group_sizes(q.shape[dim], group)
    return (q.float() * expand_groups(enc["scale"].float(), sizes, dim)
            + expand_groups(enc["min"].float(), sizes, dim)).to(dtype)


# ---- format dispatch ----

def encode(x_doc: torch.Tensor, fmt: str, role: str) -> dict:
    kind, kw = SPEC[fmt][role]
    if kind == "fp8":
        return fp8_encode(x_doc, kw["per_head"])
    if kind == "sym":
        return int_sym_encode(x_doc, kw["bits"])
    return int_asym_group_encode(x_doc, kw["bits"], kw["group"], kw["dim"])


def decode(enc: dict, fmt: str, role: str, dtype) -> torch.Tensor:
    kind, kw = SPEC[fmt][role]
    if kind == "fp8":
        return fp8_decode(enc, dtype)
    if kind == "sym":
        return int_sym_decode(enc, dtype)
    return int_asym_group_decode(enc, kw["group"], kw["dim"], dtype)


def quantize_doc(kv: DocKV, fmt: str) -> DocKV:
    """Fake quant: doc tokens encoded and decoded back to the KV dtype; shared prefix untouched.
    Returns a new DocKV (the input is never modified). 'bf16' returns the input itself."""
    if fmt == "bf16":
        return kv
    if fmt not in SPEC:
        raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")
    s = kv.shared_len

    def qdq(x: torch.Tensor, role: str) -> torch.Tensor:
        if x.shape[TOKEN_DIM] <= s:
            return x.clone()
        doc = decode(encode(x[:, s:], fmt, role), fmt, role, x.dtype)
        return torch.cat([x[:, :s], doc], dim=TOKEN_DIM)

    return DocKV([qdq(k, "k") for k in kv.keys], [qdq(v, "v") for v in kv.values], kv.next_pos, kv.shared_len)


def theory_bytes(fmt: str, n_doc: int, heads: int = 8, head_dim: int = 128, layers: int = 28) -> int:
    """Payload bytes for one doc's quantized tokens (codes + scales + mins, no file metadata).
    The 5.5 packed sizes must match this plus a known header."""
    vals = heads * n_doc * head_dim
    if fmt == "bf16":
        return layers * 2 * vals * 2
    total = 0
    for role in ("k", "v"):
        kind, kw = SPEC[fmt][role]
        if kind == "fp8":
            total += vals + (heads if kw["per_head"] else 1) * 2
        elif kind == "sym":
            total += vals * kw["bits"] // 8 + heads * n_doc * 2
        else:
            axis_len = n_doc if kw["dim"] == TOKEN_DIM else head_dim
            other = heads * (head_dim if kw["dim"] == TOKEN_DIM else n_doc)
            n_groups = len(group_sizes(axis_len, kw["group"]))
            total += math.ceil(vals * kw["bits"] / 8) + other * n_groups * 2 * 2
    return layers * total


@torch.no_grad()
def recon_stats(base: DocKV, q: DocKV) -> dict:
    """Per-layer error of the doc tokens, summable across docs:
    {k_err2, k_ref2, v_err2, v_ref2, k_maxabs, v_maxabs} as lists of length L (one .cpu() call)."""
    s = base.shared_len
    rows = []
    for A, B in ((base.keys, q.keys), (base.values, q.values)):
        x = torch.stack([a[:, s:].float() for a in A])
        d = torch.stack([b[:, s:].float() for b in B]) - x
        rows += [d.pow(2).sum(dim=(1, 2, 3)), x.pow(2).sum(dim=(1, 2, 3)), d.abs().amax(dim=(1, 2, 3))]
    k_err2, k_ref2, k_max, v_err2, v_ref2, v_max = torch.stack(rows).cpu().tolist()
    return {"k_err2": k_err2, "k_ref2": k_ref2, "v_err2": v_err2, "v_ref2": v_ref2,
            "k_maxabs": k_max, "v_maxabs": v_max}