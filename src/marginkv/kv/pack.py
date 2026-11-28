# src/marginkv/kv/pack.py
"""Packed on-disk format for document KV (Phase 5.5).

One safetensors file per doc and format holds the doc tokens only (positions shared_len..n_cached-1).
The shared system + instruction prefix is stored once per corpus (save_shared). Tensors are stacked
over layers:
  bf16:        k, v                         [L, H, n, D] bf16
  fp8_*:       {k,v}_codes [L, H, n, D] uint8 (raw e4m3fn bytes), {k,v}_scale fp16
  int8_tok:    {k,v}_codes [L, H, n, D] int8, {k,v}_scale [L, H, n, 1] fp16
  int4_*:      {k,v}_codes [L, H, n, D/2] uint8 (two codes per byte along head_dim, low nibble first),
               {k,v}_scale and {k,v}_min fp16 (shapes from quant.encode, stacked over layers)
Decoding uses quant.decode, so unpack_doc reproduces quant.quantize_doc bit for bit.
File bytes = 8 + header + payload; payload == quant.theory_bytes. Header size depends only on names,
dtypes, shapes and metadata, so predicted_file_bytes serializes zero tensors to get the exact size.
"""
from __future__ import annotations

import functools
import os

import torch
from safetensors import safe_open
from safetensors.torch import save, save_file

from marginkv.kv import quant as Q
from marginkv.kv.cache_ops import DocKV

VERSION = "marginkv-kv-v1"


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    if codes.shape[-1] % 2:
        raise ValueError("last dim must be even to pack two codes per byte")
    if codes.numel() and int(codes.max()) > 15:
        raise ValueError("int4 codes must be in 0..15")
    lo, hi = codes[..., 0::2], codes[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8).contiguous()


def unpack_nibbles(b: torch.Tensor) -> torch.Tensor:
    lo, hi = b & 0x0F, b >> 4
    return torch.stack([lo, hi], dim=-1).reshape(*b.shape[:-1], b.shape[-1] * 2)


def _dtype_name(dt: torch.dtype) -> str:
    return str(dt).split(".")[-1]


def metadata(fmt, n_doc, next_pos, shared_len, layers, heads, head_dim, dtype_name) -> dict:
    return {"version": VERSION, "format": fmt, "n_doc": str(n_doc), "next_pos": str(next_pos),
            "shared_len": str(shared_len), "layers": str(layers), "heads": str(heads),
            "head_dim": str(head_dim), "dtype": dtype_name}


def tensor_specs(fmt, n_doc, layers=28, heads=8, head_dim=128, dtype=torch.bfloat16) -> dict:
    """{name: (shape, dtype)} that doc_tensors produces for this format and doc length."""
    L, H, n, D = layers, heads, n_doc, head_dim
    sp = {}
    for role in ("k", "v"):
        if fmt == "bf16":
            sp[role] = ((L, H, n, D), dtype)
            continue
        kind, kw = Q.SPEC[fmt][role]
        if kind == "fp8":
            sp[f"{role}_codes"] = ((L, H, n, D), torch.uint8)
            sp[f"{role}_scale"] = ((L, H if kw["per_head"] else 1, 1, 1), torch.float16)
        elif kind == "sym":
            sp[f"{role}_codes"] = ((L, H, n, D), torch.int8)
            sp[f"{role}_scale"] = ((L, H, n, 1), torch.float16)
        else:
            sp[f"{role}_codes"] = ((L, H, n, D // 2), torch.uint8)
            if kw["dim"] == Q.TOKEN_DIM:
                shape = (L, H, len(Q.group_sizes(n, kw["group"])), D)
            else:
                shape = (L, H, n, len(Q.group_sizes(D, kw["group"])))
            sp[f"{role}_scale"] = (shape, torch.float16)
            sp[f"{role}_min"] = (shape, torch.float16)
    return sp


def doc_tensors(kv: DocKV, fmt: str):
    """(tensors, metadata) for one doc's packed file (CPU tensors, doc tokens only)."""
    s = kv.shared_len
    n_doc = kv.n_cached - s
    if n_doc <= 0:
        raise ValueError("doc has no tokens beyond the shared prefix")
    k0 = kv.keys[0]
    meta = metadata(fmt, n_doc, kv.next_pos, s, kv.num_layers, k0.shape[0], k0.shape[-1], _dtype_name(k0.dtype))
    t = {}
    for role, xs in (("k", kv.keys), ("v", kv.values)):
        docs = [x[:, s:].cpu() for x in xs]
        if fmt == "bf16":
            t[role] = torch.stack(docs).contiguous()
            continue
        kind, kw = Q.SPEC[fmt][role]
        encs = [Q.encode(x, fmt, role) for x in docs]
        codes = torch.stack([e["codes"] for e in encs])
        if kind == "fp8":
            codes = codes.view(torch.uint8)
        elif kind == "asym":
            assert kw["bits"] == 4, "only 4-bit asym is packed"
            codes = pack_nibbles(codes)
        t[f"{role}_codes"] = codes.contiguous()
        t[f"{role}_scale"] = torch.stack([e["scale"] for e in encs]).contiguous()
        if kind == "asym":
            t[f"{role}_min"] = torch.stack([e["min"] for e in encs]).contiguous()
    return t, meta


def _parts(tensors: dict) -> dict:
    parts = {"values": 0, "codes": 0, "scale": 0, "min": 0}
    for name, x in tensors.items():
        key = name.split("_")[-1] if "_" in name else "values"
        parts[key] += x.numel() * x.element_size()
    return parts


def pack_doc(kv: DocKV, fmt: str, path) -> dict:
    tensors, meta = doc_tensors(kv, fmt)
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    save_file(tensors, str(path), metadata=meta)
    file_bytes = os.path.getsize(path)
    parts = _parts(tensors)
    payload = sum(parts.values())
    return {"file_bytes": file_bytes, "payload_bytes": payload, "header_bytes": file_bytes - payload, "parts": parts,
            "specs": {n: (tuple(x.shape), x.dtype) for n, x in tensors.items()}}


@functools.lru_cache(maxsize=None)
def predicted_file_bytes(fmt, n_doc, shared_len, layers=28, heads=8, head_dim=128, dtype_name="bfloat16",
                         next_pos=None) -> int:
    """Exact packed file size via zero-tensor serialization.

    next_pos defaults to shared_len + n_doc (unpruned cache). Pruned caches keep their original next_pos,
    so pass it: only its digit count can change the header size.
    """
    dt = getattr(torch, dtype_name)
    zeros = {n: torch.zeros(shape, dtype=d) for n, (shape, d) in tensor_specs(fmt, n_doc, layers, heads, head_dim, dt).items()}
    np_ = shared_len + n_doc if next_pos is None else next_pos
    meta = metadata(fmt, n_doc, np_, shared_len, layers, heads, head_dim, dtype_name)
    return len(save(zeros, metadata=meta))


def unpack_doc(path, shared_keys, shared_values, device="cpu") -> DocKV:
    """Read a packed doc, decode, and prepend the given shared-prefix KV (lists of [H, s, D])."""
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
        t = {k: f.get_tensor(k) for k in f.keys()}
    if meta.get("version") != VERSION:
        raise ValueError(f"unknown packed version {meta.get('version')!r}")
    fmt, dt, L = meta["format"], getattr(torch, meta["dtype"]), int(meta["layers"])
    out = {}
    for role, shared in (("k", shared_keys), ("v", shared_values)):
        if fmt == "bf16":
            docs = list(t[role].unbind(0))
        else:
            kind, _ = Q.SPEC[fmt][role]
            codes = t[f"{role}_codes"]
            if kind == "fp8":
                codes = codes.view(torch.float8_e4m3fn)
            elif kind == "asym":
                codes = unpack_nibbles(codes)
            docs = []
            for l in range(L):
                enc = {"codes": codes[l], "scale": t[f"{role}_scale"][l]}
                if kind == "asym":
                    enc["min"] = t[f"{role}_min"][l]
                docs.append(Q.decode(enc, fmt, role, dt))
        out[role] = [torch.cat([sh.to(d.device), d], dim=1).to(device) for sh, d in zip(shared, docs)]
    return DocKV(out["k"], out["v"], int(meta["next_pos"]), int(meta["shared_len"]))


def save_shared(kv: DocKV, path) -> int:
    """Store the shared prefix (kv must hold exactly shared_len tokens) once per corpus."""
    if kv.n_cached != kv.shared_len:
        raise ValueError("shared KV must contain exactly shared_len tokens")
    k0 = kv.keys[0]
    tensors = {"k": torch.stack([k.cpu() for k in kv.keys]).contiguous(),
               "v": torch.stack([v.cpu() for v in kv.values]).contiguous()}
    meta = {"version": VERSION, "kind": "shared", "shared_len": str(kv.shared_len),
            "layers": str(kv.num_layers), "dtype": _dtype_name(k0.dtype)}
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    save_file(tensors, str(path), metadata=meta)
    return os.path.getsize(path)


def load_shared(path):
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
        k, v = f.get_tensor("k"), f.get_tensor("v")
    return list(k.unbind(0)), list(v.unbind(0)), meta