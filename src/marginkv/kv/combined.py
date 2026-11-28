# src/marginkv/kv/combined.py
"""Phase 7: combined compression of a document KV cache.

Order is fixed: prune doc tokens first (kv.prune), then quantize the kept
doc tokens (kv.quant). The shared system + instruction prefix stays BF16 and
next_pos is unchanged, so query positions start at the original doc length.
KIVI key groups therefore form over kept tokens (ceil(kept / 32) groups).
"""
from __future__ import annotations

from marginkv.kv import prune as P
from marginkv.kv import quant as Q

ORDER = "prune_then_quant"


def arm_name(method: str, retention: float, fmt: str) -> str:
    """uniform, 0.75, int4_kivi -> 'uniform_r75_int4_kivi' (Phase 6 style prefix)."""
    return f"{method}_r{round(retention * 100)}_{fmt}"


class CombinedTransform:
    """Doc-aware sweep transform: fn(kv, doc_id) -> DocKV.

    A class (not a closure) so `infos` can be a live view of the prune
    transform's per-doc retention info, which the sweep reads.
    """

    wants_doc = True

    def __init__(self, method, retention, fmt, importance=None, seed=None):
        if fmt not in Q.FORMATS:
            raise ValueError(f"unknown format {fmt!r}; expected one of {Q.FORMATS}")
        self.prune = P.make_transform(method, retention, importance=importance, seed=seed)
        self.method = method
        self.retention = retention
        self.fmt = fmt
        self.order = ORDER
        self.name = arm_name(method, retention, fmt)

    @property
    def infos(self):
        return self.prune.infos

    def __call__(self, kv, doc_id):
        pruned = self.prune(kv, doc_id)
        return Q.quantize_doc(pruned, self.fmt)


def make_combined(method, retention, fmt, importance=None, seed=None) -> CombinedTransform:
    return CombinedTransform(method, retention, fmt, importance=importance, seed=seed)