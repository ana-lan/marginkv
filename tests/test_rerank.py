# tests/test_rerank.py
"""Plain-assert tests for marginkv.eval.rerank. Run: python tests/test_rerank.py"""
from __future__ import annotations

import math

from marginkv.config import load_config
from marginkv.eval.metrics import evaluate_ranked
from marginkv.eval.rerank import evaluate_rerank, identity_scores, rerank_merge
from marginkv.utils.io import load_json


def _cands(n=5):
    return {"q1": [[f"d{i}", float(10 - i)] for i in range(1, n + 1)]}


def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def test_identity_reproduces_bm25_order():
    c = _cands(6)
    out = rerank_merge(c, identity_scores(c, 3), depth=3)
    assert out["q1"] == [d for d, _ in c["q1"]], out


def test_bm25_scores_with_ties_keep_bm25_order():
    # d2, d3 tie on BM25 score; candidates already broke the tie. Feeding BM25 scores back must not reorder.
    c = {"q1": [["d1", 5.0], ["d2", 3.0], ["d3", 3.0], ["d4", 1.0]]}
    s = {"q1": {d: sc for d, sc in c["q1"][:3]}}
    assert rerank_merge(c, s, depth=3)["q1"] == ["d1", "d2", "d3", "d4"]


def test_hand_swap_and_metrics():
    c = _cands(5)
    s = {"q1": {"d1": 0.2, "d2": 0.1, "d3": 0.9}}
    out = rerank_merge(c, s, depth=3)
    assert out["q1"] == ["d3", "d1", "d2", "d4", "d5"], out
    qrels = {"q1": {"d3": 1}}
    bm25 = evaluate_ranked({"q1": [d for d, _ in c["q1"]]}, qrels)["mean"]
    rr = evaluate_ranked(out, qrels)["mean"]
    assert abs(bm25["mrr@10"] - 1 / 3) < 1e-9, bm25
    assert abs(rr["mrr@10"] - 1.0) < 1e-9 and abs(rr["ndcg@10"] - 1.0) < 1e-9, rr


def test_score_ties_fall_back_to_bm25_rank():
    c = _cands(4)
    s = {"q1": {"d1": 0.5, "d2": 0.9, "d3": 0.9}}
    assert rerank_merge(c, s, depth=3)["q1"] == ["d2", "d3", "d1", "d4"]
    s = {"q1": {"d1": 1.0, "d2": 1.0, "d3": 1.0}}  # all tied: BM25 order
    assert rerank_merge(c, s, depth=3)["q1"] == ["d1", "d2", "d3", "d4"]


def test_tail_untouched():
    c = _cands(6)
    s = {"q1": {"d1": -5.0, "d2": -6.0, "d3": -7.0}}  # low scores never push head docs into the tail
    assert rerank_merge(c, s, depth=3)["q1"][3:] == ["d4", "d5", "d6"]


def test_bad_inputs_raise():
    c = _cands(4)
    assert _raises(lambda: rerank_merge(c, {"q1": {"d1": 1.0, "d2": 0.5}}, 3))                       # missing doc
    assert _raises(lambda: rerank_merge(c, {"q1": {"d1": 1, "d2": 1, "d3": 1, "d4": 1}}, 3))        # extra doc
    assert _raises(lambda: rerank_merge(c, {"q1": {"d1": math.nan, "d2": 1, "d3": 1}}, 3))         # NaN
    assert _raises(lambda: rerank_merge(c, {}, 3))                                                  # missing query
    assert _raises(lambda: rerank_merge(c, identity_scores(_cands(4), 3), depth=5))                # depth > pool


def test_real_identity_matches_saved_bm25():
    # Identity scores on the real candidates must reproduce Phase 1 BM25 metrics exactly.
    from marginkv.data.beir_loader import load_beir

    cfg = load_config()
    saved = load_json(cfg.paths.results_dir / "metrics" / "bm25.json")["datasets"]
    for name in cfg.datasets:
        path = cfg.paths.results_dir / "candidates" / f"{name}.json"
        if not path.exists():
            print(f"  skip {name}: no candidates file")
            continue
        cands = load_json(path)["candidates"]
        ds = load_beir(name, split=cfg.split, data_dir=cfg.paths.data_dir)
        m = evaluate_rerank(cands, identity_scores(cands, cfg.rerank.depth), ds.qrels, cfg.rerank.depth)
        for key in ("ndcg@10", "mrr@10", "recall@100"):
            assert abs(m["mean"][key] - saved[name]["mean"][key]) < 1e-12, (name, key, m["mean"][key])
        d = m["diagnostics"]
        assert d["top1_changed_frac"] == 0 and d["mean_abs_rank_shift"] == 0, d
        print(f"  {name}: identity rerank == BM25 "
              f"(nDCG@10 {m['mean']['ndcg@10']:.4f}, R@100 {m['mean']['recall@100']:.4f})")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(tests)} rerank tests passed")