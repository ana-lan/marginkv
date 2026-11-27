# src/marginkv/kv/prune.py
"""Query-independent token pruning of document KV caches (Phase 6).

Only doc tokens are pruned; the shared system + instruction prefix is never
touched. Kept tokens keep their original positions (RoPE is already applied
to the cached keys) and next_pos is unchanged, so query positions still start
at the original document length.
"""
from __future__ import annotations

import hashlib
import math

import torch

from marginkv.kv.cache_ops import DocKV

METHODS = ("uniform", "boundary", "attn_a", "attn_b", "random")
RETENTIONS = (0.25, 0.50, 0.75)
FIRST_KEEP = 1       # mandatory first doc tokens
RESERVE = 8          # mandatory last doc tokens (incl. the seam token; importance undefined there)
BOUNDARY_FIRST = 4   # boundary method: first tokens kept before the recent window
DIAGNOSTIC_METHODS = ("attn_h2o",)


def doc_len(kv: DocKV) -> int:
    """Original doc token count T (correct for pruned caches too)."""
    return kv.next_pos - kv.shared_len


def kept_doc_tokens(kv: DocKV) -> int:
    return kv.n_cached - kv.shared_len


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def mandatory_idx(T: int, first: int = FIRST_KEEP, reserve: int = RESERVE) -> list[int]:
    head = range(min(first, T))
    tail = range(max(0, T - reserve), T)
    return sorted(set(head) | set(tail))


def budget(T: int, retention: float, n_mandatory: int) -> tuple[int, int]:
    """(target, kept): target = half-up(retention * T); kept never below the mandatory set."""
    target = round_half_up(retention * T)
    return target, min(T, max(target, n_mandatory))


def doc_seed(seed: int, doc_id: str) -> int:
    """Stable per-doc seed for the random method."""
    h = hashlib.sha256(f"{seed}-{doc_id}".encode()).digest()
    return int.from_bytes(h[:8], "little") & (2**63 - 1)


def importance_from_attn(attn: torch.Tensor, shared_len: int, weighted: bool) -> torch.Tensor:
    """Token importance from one doc prefix's attention.

    attn: [S, S] attention probabilities averaged over heads and layers,
    S = shared_len + T (rows = queries j, cols = keys i, absolute positions).
    Doc-relative i in [0, T):
      (a) I_i = sum_{j > i} A(j, i) / (T - i - 1)
      (b) same with each A(j, i) times (shared_len + j + 1), the absolute
          position + 1 (uniform causal attention gives I_i = 1 for every i).
    Rows and cols restricted to doc tokens. Returns float64 [T] on CPU;
    the last token has no later rows and is NaN.
    """
    attn = attn.detach().cpu().double()  # move first: MPS has no float64
    S = attn.shape[0]
    if attn.shape != (S, S) or S <= shared_len:
        raise ValueError(f"bad attn shape {tuple(attn.shape)} for shared_len {shared_len}")
    s, T = shared_len, S - shared_len
    D = attn[s:, s:] * torch.tril(torch.ones(T, T, dtype=torch.bool), diagonal=-1)
    if weighted:
        w = torch.arange(T, dtype=torch.float64) + s + 1
        D = D * w[:, None]
    sums = D.sum(0)
    counts = torch.arange(T - 1, -1, -1, dtype=torch.float64)  # T - i - 1
    out = torch.full((T,), float("nan"), dtype=torch.float64)
    m = counts > 0
    out[m] = sums[m] / counts[m]
    return out


def select_keep(method: str, T: int, retention: float, *, importance=None, seed: int | None = None,
                first: int = FIRST_KEEP, reserve: int = RESERVE) -> tuple[torch.Tensor, dict]:
    """Doc-relative indices to keep (sorted LongTensor) and retention info."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method}")
    if not 0 < retention <= 1:
        raise ValueError(f"retention must be in (0, 1], got {retention}")
    if T < 1:
        raise ValueError("empty doc")
    mand = mandatory_idx(T, first, reserve)
    target, k = budget(T, retention, len(mand))
    mset = set(mand)
    free = [i for i in range(T) if i not in mset]
    n_free = k - len(mand)

    chosen: list[int] = []
    if n_free > 0:
        if method == "uniform":
            step = len(free) / n_free  # >= 1, so picks are distinct
            chosen = [free[int((j + 0.5) * step)] for j in range(n_free)]
        elif method == "boundary":
            order = [i for i in free if i < BOUNDARY_FIRST] + sorted((i for i in free if i >= BOUNDARY_FIRST), reverse=True)
            chosen = order[:n_free]
        elif method in ("attn_a", "attn_b"):
            if importance is None:
                raise ValueError(f"{method} needs importance")
            imp = torch.as_tensor(importance, dtype=torch.float64).cpu()
            if imp.shape != (T,):
                raise ValueError(f"importance shape {tuple(imp.shape)} != ({T},)")
            vals = imp[free]
            if not torch.isfinite(vals).all():
                raise ValueError("non-finite importance on a prunable token")
            order = torch.sort(-vals, stable=True).indices[:n_free].tolist()  # desc, ties -> lower index
            chosen = [free[o] for o in order]
        else:  # random
            if seed is None:
                raise ValueError("random needs seed")
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(len(free), generator=g)[:n_free].tolist()
            chosen = [free[p] for p in perm]

    keep = sorted(mset | set(chosen))
    assert len(keep) == k, (len(keep), k)
    info = {
        "doc_tokens": T,
        "target": target,
        "kept": k,
        "mandatory": len(mand),
        "target_retention": retention,
        "actual_retention": k / T,
        "forced": k > target,
    }
    return torch.tensor(keep, dtype=torch.long), info


def prune_doc(kv: DocKV, keep) -> DocKV:
    """New DocKV with the shared prefix plus the kept doc tokens (doc-relative indices).

    Input must be unpruned. next_pos and shared_len are unchanged. The input is not modified.
    """
    if kv.n_cached != kv.next_pos:
        raise ValueError(f"prune_doc expects an unpruned cache (n_cached {kv.n_cached} != next_pos {kv.next_pos})")
    T = doc_len(kv)
    keep = torch.as_tensor(keep, dtype=torch.long).cpu()
    if keep.ndim != 1 or len(keep) == 0:
        raise ValueError("keep must be a non-empty 1-D index list")
    if keep[0] < 0 or keep[-1] >= T or (len(keep) > 1 and not torch.all(keep[1:] > keep[:-1])):
        raise ValueError("keep must be sorted, unique, within [0, T)")
    dev = kv.keys[0].device
    idx = torch.cat([torch.arange(kv.shared_len), kv.shared_len + keep]).to(dev)
    keys = [k.index_select(1, idx) for k in kv.keys]
    values = [v.index_select(1, idx) for v in kv.values]
    return DocKV(keys=keys, values=values, next_pos=kv.next_pos, shared_len=kv.shared_len)


def importance_h2o(imp_a: torch.Tensor) -> torch.Tensor:
    """Unnormalized accumulated attention sum_{j>i} A(j,i) (H2O-style), recovered exactly from (a).

    (a)_i = sum / (T - i - 1), so sum = (a)_i * (T - i - 1). Last token stays NaN.
    """
    imp_a = torch.as_tensor(imp_a, dtype=torch.float64).cpu()
    T = len(imp_a)
    counts = torch.arange(T - 1, -1, -1, dtype=torch.float64)
    out = imp_a * counts
    out[-1] = float("nan")
    return out


def make_transform(method: str, retention: float, importance: dict | None = None, seed: int | None = None):
    """Doc-aware pruning transform fn(kv, doc_id) -> DocKV for kv.sweep.

    importance: {doc_id: tensor [T]} for attn_* methods. seed: base seed for random
    (per-doc seed = doc_seed(seed, doc_id)). Per-doc retention info is collected in fn.infos.
    uniform at retention 1.0 is the keep-all check arm.
    """
    name = method if method in METHODS else None
    if method not in METHODS + DIAGNOSTIC_METHODS:
        raise ValueError(f"unknown method {method}")
    if method.startswith("attn") and importance is None:
        raise ValueError(f"{method} needs importance")
    if method == "random" and seed is None:
        raise ValueError("random needs seed")
    sel = "attn_a" if method == "attn_h2o" else name  # selection is identical for every attn_* score
    infos: dict = {}

    def fn(kv: DocKV, doc: str) -> DocKV:
        T = doc_len(kv)
        kw = {}
        if method.startswith("attn"):
            kw["importance"] = importance[doc]
        if method == "random":
            kw["seed"] = doc_seed(seed, doc)
        keep, info = select_keep(sel, T, retention, **kw)
        infos[doc] = info
        return prune_doc(kv, keep)

    fn.infos = infos
    fn.wants_doc = True
    fn.method, fn.retention = method, retention
    return fn