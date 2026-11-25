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