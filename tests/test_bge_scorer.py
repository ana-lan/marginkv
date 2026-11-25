# tests/test_bge_scorer.py
"""Checks for the bge-reranker-v2-m3 baseline. Loads the model (MPS bf16 + CPU fp32). Writes nothing.
Run: python tests/test_bge_scorer.py
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from marginkv.config import load_config
from marginkv.data.beir_loader import load_beir
from marginkv.models.bge_reranker import encode, load_model, load_tokenizer, pair_ids, score_batch, score_sequences
from marginkv.retrieval.bm25 import doc_text
from marginkv.utils.device import get_device, get_dtype
from marginkv.utils.io import load_json
from marginkv.utils.seed import set_seed

# Reference values from the model card (fp16 FlagReranker): roughly -8.19 and +5.26.
CARD = [("what is panda?", "hi"),
        ("what is panda?", "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or "
                           "simply panda, is a bear species endemic to China.")]


def _flips(a, b):
    n = len(a)
    return sum((a[i] - a[j]) * (b[i] - b[j]) < 0 for i in range(n) for j in range(i + 1, n))


def main() -> None:
    cfg = load_config()
    set_seed(cfg.seed)
    name, rev = cfg.models.bge.name, cfg.models.bge.revision
    max_doc, depth = cfg.rerank.max_doc_tokens, cfg.rerank.depth
    tok = load_tokenizer(name, rev)
    pad = tok.pad_token_id
    print(f"tokenizer {type(tok).__name__}: cls={tok.cls_token_id} sep={tok.sep_token_id} pad={pad}")
    assert pad == 1, "XLM-R pad id should be 1 (positions depend on it)"

    # 1) Tokenization parity on every SciFact reranked pair (exact if untruncated, prefix if truncated).
    ds = load_beir("scifact", split=cfg.split, data_dir=cfg.paths.data_dir)
    cands = load_json(cfg.paths.results_dir / "candidates" / "scifact.json")
    pairs = [(q, d) for q in cands["query_order"] for d, _ in cands["candidates"][q][:depth]]
    q_ids = dict(zip(cands["query_order"], encode(tok, [ds.queries[q] for q in cands["query_order"]])))
    docs = sorted({d for _, d in pairs})
    d_full = dict(zip(docs, encode(tok, [doc_text(ds.corpus[d]) for d in docs])))
    ref = tok([ds.queries[q] for q, _ in pairs], [doc_text(ds.corpus[d]) for _, d in pairs])["input_ids"]
    ok_full = ok_trunc = n_full = n_trunc = 0
    for (q, d), r in zip(pairs, ref):
        ours = pair_ids(tok, q_ids[q], d_full[d][:max_doc])
        if len(d_full[d]) <= max_doc:
            n_full += 1
            ok_full += ours == r
        else:
            n_trunc += 1
            ok_trunc += r[:len(ours) - 1] == ours[:-1]  # drop our closing </s>, compare as prefix
    trunc_frac = sum(len(v) > max_doc for v in d_full.values()) / len(d_full)
    print(f"\n[1] pair parity: full {ok_full}/{n_full}, truncated (prefix) {ok_trunc}/{n_trunc}; "
          f"docs truncated at {max_doc}: {trunc_frac:.1%}")
    assert ok_full == n_full and ok_trunc == n_trunc, "bge pair ids differ from tokenizer(query, doc)"

    dev, dtype, cpu = get_device(cfg.runtime.device), get_dtype(cfg.runtime.dtype), torch.device("cpu")
    m_dev = load_model(name, rev, dev, dtype)
    m_cpu = load_model(name, rev, cpu, torch.float32)

    # 2) Card pairs, CPU fp32: our split path vs the plain model forward.
    print("\n[2] model card pairs (CPU fp32)")
    for q, d in CARD:
        ids = pair_ids(tok, encode(tok, [q])[0], encode(tok, [d])[0])
        assert ids == tok(q, d)["input_ids"]
        ours = score_batch(m_cpu, [ids], pad, cpu)["logit"][0]
        with torch.inference_mode():
            plain = m_cpu(input_ids=torch.tensor([ids])).logits[0, 0].item()
        print(f"  {d[:40]!r:44s} logit={ours:+.4f}  |ours-plain|={abs(ours - plain):.2e}")
        assert abs(ours - plain) < 1e-4
    s_hi = score_batch(m_cpu, [pair_ids(tok, encode(tok, [CARD[0][0]])[0], encode(tok, [CARD[0][1]])[0])], pad, cpu)
    s_panda = score_batch(m_cpu, [pair_ids(tok, encode(tok, [CARD[1][0]])[0], encode(tok, [CARD[1][1]])[0])], pad, cpu)
    assert s_hi["logit"][0] < -4 and s_panda["logit"][0] > 3, "card pairs far from reference (-8.19, +5.26)"

    # 3) Real pairs: batched vs single on device, device vs CPU fp32.
    sub = [(q, d) for q in cands["query_order"][:2] for d, _ in cands["candidates"][q][:8]]
    seqs = [pair_ids(tok, q_ids[q], d_full[d][:max_doc]) for q, d in sub]
    b = score_sequences(m_dev, seqs, pad, batch_size=8, device=dev)
    single = [score_batch(m_dev, [s], pad, dev)["logit"][0] for s in seqs]
    assert all(torch.isfinite(torch.tensor(b["logit"]))), "non-finite bge scores"
    d_bs = max(abs(x - y) for x, y in zip(b["logit"], single))
    c = score_sequences(m_cpu, seqs, pad, batch_size=8, device=cpu)
    d_dc = max(abs(x - y) for x, y in zip(b["logit"], c["logit"]))
    flips = sum(_flips(b["logit"][i:i + 8], c["logit"][i:i + 8]) for i in (0, 8))
    print(f"\n[3] {len(seqs)} real pairs: batched vs single ({dev}) max |d| = {d_bs:.4f}; "
          f"{dev} {dtype} vs CPU fp32 max |d| = {d_dc:.4f}; flips = {flips}/56")
    assert d_bs < 0.25 and d_dc < 0.5, "bge drift beyond bf16 noise"
    print("\n  qid  doc_id      len   logit")
    for (q, d), s, lo in zip(sub, seqs, b["logit"]):
        print(f"  {q:>3}  {d:>10}  {len(s):4d}  {lo:+7.3f}")
    print("\nall bge scorer checks passed")


if __name__ == "__main__":
    main()