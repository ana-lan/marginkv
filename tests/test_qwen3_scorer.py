# tests/test_qwen3_scorer.py
"""Checks for the Qwen3 query-first scorer. Loads the model (MPS bf16 + CPU fp32). Writes nothing.
Run: python tests/test_qwen3_scorer.py
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import time

import torch

from marginkv.config import load_config
from marginkv.models.qwen3_reranker import (
    DEFAULT_INSTRUCTION, NO_ID, YES_ID, build_pieces, doc_piece_text, encode, load_model,
    load_tokenizer, official_query_first_ids, query_first_ids, query_text_qf, score_batch,
    score_sequences,
)
from marginkv.utils.device import get_device, get_dtype
from marginkv.utils.io import load_pt
from marginkv.utils.seed import set_seed

CARD_QUERIES = ["What is the capital of China?", "Explain gravity"]
CARD_DOCS = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical "
    "objects and is responsible for the movement of planets around the sun.",
]


@torch.inference_mode()
def _official_single(model, ids, device) -> float:
    """Model card style forward: one unpadded sequence, no position_ids, full logits."""
    last = model(input_ids=torch.tensor([ids], device=device)).logits[0, -1].float()
    return (last[YES_ID] - last[NO_ID]).item()


def _flips(a: list[float], b: list[float]) -> int:
    """Pairwise order disagreements between two score lists."""
    n = len(a)
    return sum((a[i] - a[j]) * (b[i] - b[j]) < 0 for i in range(n) for j in range(i + 1, n))


def main() -> None:
    cfg = load_config()
    set_seed(cfg.seed)
    name, rev = cfg.models.qwen3.name, cfg.models.qwen3.revision
    tok = load_tokenizer(name, rev)
    pieces = build_pieces(tok, DEFAULT_INSTRUCTION)

    dev = get_device(cfg.runtime.device)
    dtype = get_dtype(cfg.runtime.dtype)
    cpu = torch.device("cpu")
    print(f"device={dev} dtype={dtype}")
    m_dev = load_model(name, rev, dev, dtype)
    m_cpu = load_model(name, rev, cpu, torch.float32)

    # 1) Model card pairs: our ids == official ids; our scorer == official forward (CPU fp32, unpadded).
    grid = [(qi, di) for qi in range(2) for di in range(2)]
    ours_ids, ref_ids = [], official_query_first_ids(
        tok, DEFAULT_INSTRUCTION, [CARD_QUERIES[q] for q, _ in grid], [CARD_DOCS[d] for _, d in grid])
    for q, d in grid:
        qi = encode(tok, [query_text_qf(CARD_QUERIES[q])])[0]
        di = encode(tok, [doc_piece_text(CARD_DOCS[d])])[0]
        ours_ids.append(query_first_ids(pieces, qi, di))
    assert ours_ids == ref_ids, "card pair ids differ from the model card's tokenization"

    print("\n[1] model card pairs (CPU fp32)")
    for (q, d), ids in zip(grid, ours_ids):
        ours = score_batch(m_cpu, [ids], cpu)
        ref = _official_single(m_cpu, ids, cpu)
        diff = abs(ours["log_odds"][0] - ref)
        print(f"  q{q} d{d}  P(yes)={ours['p_yes'][0]:.4f}  log_odds={ours['log_odds'][0]:+.3f}  |ours-official|={diff:.2e}")
        assert diff < 1e-3, f"scorer differs from official forward: {diff}"
        if q == d:
            assert ours["p_yes"][0] > 0.9, "relevant card pair scored low"
        else:
            assert ours["p_yes"][0] < 0.1, "irrelevant card pair scored high"

    # 2) Real SciFact pairs, mixed lengths: batched vs single on the run device, then device vs CPU fp32.
    a = load_pt(cfg.paths.artifacts_dir / "tokenized" / "scifact_qwen3.pt")
    pairs = [(q, d) for q in a["query_order"][:2] for d in a["rerank_candidates"][q][:8]]
    seqs = [query_first_ids(a["pieces"], a["query_qf"][q], a["doc"][d]) for q, d in pairs]
    lens = [len(s) for s in seqs]
    print(f"\n[2] {len(seqs)} real SciFact pairs, lengths {min(lens)}..{max(lens)}")

    t0 = time.perf_counter()
    batched = score_sequences(m_dev, seqs, batch_size=8, device=dev)
    t_batch = time.perf_counter() - t0
    single = [score_batch(m_dev, [s], dev)["log_odds"][0] for s in seqs]
    bl = batched["log_odds"]
    assert all(torch.isfinite(torch.tensor(bl))), "non-finite scores in batched run"
    d_bs = max(abs(x - y) for x, y in zip(bl, single))
    print(f"  batched vs single ({dev}): max |d log_odds| = {d_bs:.4f}")
    assert d_bs < 0.25, "batched scoring differs from single beyond bf16 noise: check padding/positions"

    ref_cpu = score_sequences(m_cpu, seqs, batch_size=8, device=cpu)
    d_lo = max(abs(x - y) for x, y in zip(bl, ref_cpu["log_odds"]))
    d_p = max(abs(x - y) for x, y in zip(batched["p_yes"], ref_cpu["p_yes"]))
    flips = sum(_flips(bl[i:i + 8], ref_cpu["log_odds"][i:i + 8]) for i in (0, 8))
    print(f"  {dev} {dtype} vs CPU fp32: max |d log_odds| = {d_lo:.4f}, max |d P(yes)| = {d_p:.4f}, "
          f"pairwise order flips = {flips}/56")
    assert d_p < 0.05, "device vs CPU drift too large"

    print("\n  qid  doc_id      len  P(yes)  log_odds")
    for (q, d), n, p, lo in zip(pairs, lens, batched["p_yes"], bl):
        print(f"  {q:>3}  {d:>10}  {n:4d}  {p:.4f}  {lo:+7.3f}")
    print(f"\n  timing (rough, includes warmup): {len(seqs)} pairs in {t_batch:.2f}s on {dev}")
    print("\nall qwen3 scorer checks passed")


if __name__ == "__main__":
    main()