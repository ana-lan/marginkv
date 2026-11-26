# src/marginkv/kv/cache_ops.py
"""Document KV cache: build once per doc, score many queries on it (Phase 3.3).

DocKV holds plain per-layer tensors, not an HF cache object, so later phases can quantize or
prune them directly. Every scoring call wraps the tensors in a fresh DynamicCache, because HF
caches grow in place during a forward pass; the stored DocKV is never modified.

Layout of each row in a cached batch (both segments left padded, pads masked):
    [pad .. | doc KV (n_cached)] [pad .. | new tokens (tag + query + suffix)]
Cached keys already carry RoPE at their original positions, so new tokens get
position_ids = next_pos + arange(len). The causal mask works on slot indices (cache_position),
so left padding inside the cache doesn't shift any RoPE position.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import DynamicCache

from marginkv.models.qwen3_reranker import NO_ID, PAD_ID, YES_ID, collate, doc_first_ids, doc_first_seam_ids


@dataclass
class DocKV:
    keys: list[torch.Tensor]     # per layer [kv_heads, n_cached, head_dim], RoPE applied
    values: list[torch.Tensor]   # per layer [kv_heads, n_cached, head_dim]
    next_pos: int                # position id of the first token after the cache (original prefix length)
    shared_len: int              # leading tokens identical for every doc (system + instruction)

    @property
    def n_cached(self) -> int:
        return self.keys[0].shape[-2]

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (*self.keys, *self.values))

    def to(self, device) -> "DocKV":
        return DocKV([k.to(device) for k in self.keys], [v.to(device) for v in self.values],
                     self.next_pos, self.shared_len)

    def clone(self) -> "DocKV":
        return DocKV([k.clone() for k in self.keys], [v.clone() for v in self.values],
                     self.next_pos, self.shared_len)


def shared_prefix_len(pieces: dict) -> int:
    return len(pieces["prefix"]) + len(pieces["df_head"])


def split_doc_first(pieces: dict, doc: list[int], q_df: list[int], variant: str = "B"):
    """Return (cacheable prefix ids, new ids) for one doc-first pair.
    Variant B expects `doc` to be the seam doc piece and `pieces` to include df_mid_seam."""
    prefix = pieces["prefix"] + pieces["df_head"] + doc
    if variant == "A":
        new = pieces["df_mid"] + q_df + pieces["suffix"]
        full = doc_first_ids(pieces, doc, q_df)
    elif variant == "B":
        new = pieces["df_mid_seam"] + q_df + pieces["suffix"]
        full = doc_first_seam_ids(pieces, doc, q_df)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    assert prefix + new == full, "prefix + new must equal the full doc-first prompt"
    return prefix, new


def _yes_no(model, h: torch.Tensor) -> dict[str, list[float]]:
    """Same head as qwen3_reranker.score_batch: fp32 yes/no logits from the final hidden state."""
    w = model.lm_head.weight[[NO_ID, YES_ID]].float()
    logits = h.float() @ w.T
    no, yes = logits[:, 0], logits[:, 1]
    lo = yes - no
    return {"logit_yes": yes.cpu().tolist(), "logit_no": no.cpu().tolist(),
            "log_odds": lo.cpu().tolist(), "p_yes": torch.sigmoid(lo).cpu().tolist()}


@torch.inference_mode()
def build_doc_kvs(model, prefixes: list[list[int]], device, shared_len: int) -> list[DocKV]:
    """One batched forward over left-padded prefixes; slice each row's real tokens out of the cache."""
    ids, mask, pos = collate(prefixes, PAD_ID, device)
    out = model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
    layers = out.past_key_values.layers
    assert len(layers) == model.config.num_hidden_layers, f"got {len(layers)} cache layers"
    T = ids.shape[1]
    kvs = []
    for b, p in enumerate(prefixes):
        n = len(p)
        ks = [lay.keys[b, :, T - n:, :].clone() for lay in layers]   # clone: don't pin the batch tensor
        vs = [lay.values[b, :, T - n:, :].clone() for lay in layers]
        kvs.append(DocKV(ks, vs, next_pos=n, shared_len=shared_len))
    return kvs


@torch.inference_mode()
def score_on_caches(model, kvs: list[DocKV], new_seqs: list[list[int]], device) -> dict[str, list[float]]:
    """Score row b = new_seqs[b] on top of kvs[b]. Rows may use different docs (KV left padded to the
    longest). Returns the same keys as score_batch, in input order. kvs must already be on `device`."""
    if not kvs or len(kvs) != len(new_seqs) or any(len(s) == 0 for s in new_seqs):
        raise ValueError("need equal, non-empty lists of caches and new sequences")
    B = len(kvs)
    P = max(kv.n_cached for kv in kvs)
    N = max(len(s) for s in new_seqs)
    ref = kvs[0].keys[0]
    H, D, dt = ref.shape[0], ref.shape[-1], ref.dtype

    cache = DynamicCache()
    for layer in range(kvs[0].num_layers):
        k = torch.zeros(B, H, P, D, dtype=dt, device=device)
        v = torch.zeros(B, H, P, D, dtype=dt, device=device)
        for b, kv in enumerate(kvs):
            n = kv.n_cached
            k[b, :, P - n:] = kv.keys[layer]
            v[b, :, P - n:] = kv.values[layer]
        cache.update(k, v, layer)   # fresh cache per call; the DocKV tensors are copied, never mutated

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
    return _yes_no(model, out.last_hidden_state[:, -1, :])


@dataclass
class KVChunk:
    """Several DocKVs stacked once (Phase 5.1). keys/values: [L, n_docs, H, P, D], each doc left padded
    to P = longest doc in the chunk. Scoring slices to the batch's own longest doc before gathering,
    so the tensors fed to the model equal what score_on_caches builds for the same batch."""
    keys: torch.Tensor
    values: torch.Tensor
    n_cached: list[int]
    next_pos: list[int]
    shared_len: int

    @property
    def n_docs(self) -> int:
        return self.keys.shape[1]

    @property
    def P(self) -> int:
        return self.keys.shape[-2]

    def nbytes(self) -> int:
        return 2 * self.keys.numel() * self.keys.element_size()


def stack_chunk(kvs: list[DocKV]) -> KVChunk:
    """Stack DocKVs (already on their device, same dtype) into one left-padded chunk.
    Compress each DocKV *before* stacking: padding zeros must not enter quantization statistics."""
    if not kvs:
        raise ValueError("need at least one DocKV")
    k0 = kvs[0].keys[0]
    L, H, D, dt, dev = kvs[0].num_layers, k0.shape[0], k0.shape[-1], k0.dtype, k0.device
    for kv in kvs:
        if kv.num_layers != L or kv.shared_len != kvs[0].shared_len:
            raise ValueError("DocKVs differ in layers or shared_len")
        if kv.keys[0].dtype != dt or kv.keys[0].device != dev:
            raise ValueError("DocKVs differ in dtype or device")
    P = max(kv.n_cached for kv in kvs)
    keys = torch.zeros(L, len(kvs), H, P, D, dtype=dt, device=dev)
    values = torch.zeros_like(keys)
    for i, kv in enumerate(kvs):
        n = kv.n_cached
        keys[:, i, :, P - n:] = torch.stack(kv.keys)      # [L, H, n, D]
        values[:, i, :, P - n:] = torch.stack(kv.values)
    return KVChunk(keys, values, [kv.n_cached for kv in kvs], [kv.next_pos for kv in kvs], kvs[0].shared_len)


@torch.inference_mode()
def score_on_chunk(model, chunk: KVChunk, rows: list[int], new_seqs: list[list[int]], device) -> dict[str, list[float]]:
    """Score new_seqs[b] on chunk doc rows[b]. Same result as score_on_caches on the same batch
    (identical tensors and shapes), with two gathers instead of per-layer, per-row copies.
    The chunk must already be on `device`; it is never modified."""
    if not rows or len(rows) != len(new_seqs) or any(len(s) == 0 for s in new_seqs):
        raise ValueError("need equal, non-empty lists of rows and new sequences")
    if any(r < 0 or r >= chunk.n_docs for r in rows):
        raise IndexError("row out of range for this chunk")
    if chunk.keys.device.type != torch.device(device).type:
        raise ValueError("chunk must be on the scoring device")
    B = len(rows)
    n_c = [chunk.n_cached[r] for r in rows]
    P = max(n_c)                      # batch's own longest doc, same as score_on_caches
    N = max(len(s) for s in new_seqs)
    start = chunk.P - P               # every row's real tokens sit in the last n <= P slots

    idx = torch.tensor(rows, dtype=torch.long, device=chunk.keys.device)
    k = chunk.keys[:, :, :, start:, :].index_select(1, idx)     # [L, B, H, P, D], one contiguous copy
    v = chunk.values[:, :, :, start:, :].index_select(1, idx)
    cache = DynamicCache()
    for layer in range(k.shape[0]):
        cache.update(k[layer], v[layer], layer)

    ids = torch.full((B, N), PAD_ID, dtype=torch.long)
    mask = torch.zeros((B, P + N), dtype=torch.long)
    pos = torch.zeros((B, N), dtype=torch.long)
    for b, (r, s) in enumerate(zip(rows, new_seqs)):
        m, n = len(s), chunk.n_cached[r]
        ids[b, N - m:] = torch.tensor(s, dtype=torch.long)
        mask[b, P - n:P] = 1
        mask[b, P + N - m:] = 1
        pos[b, N - m:] = torch.arange(chunk.next_pos[r], chunk.next_pos[r] + m)

    out = model.model(input_ids=ids.to(device), attention_mask=mask.to(device),
                      position_ids=pos.to(device), past_key_values=cache, use_cache=True)
    return _yes_no(model, out.last_hidden_state[:, -1, :])