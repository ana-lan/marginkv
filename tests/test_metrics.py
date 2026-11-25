# tests/test_metrics.py
"""Hand-computed cases. Runs under pytest or plain `python tests/test_metrics.py`."""
import math

import pytrec_eval

from marginkv.eval.metrics import evaluate_ranked

L3 = 1 / math.log2(3)  # discount at rank 2


def _close(a, b, tol=1e-6):
    assert abs(a - b) < tol, f"{a} != {b}"


def test_binary_rank2():
    r = evaluate_ranked({"q": ["d2", "d1", "d3"]}, {"q": {"d1": 1}})["per_query"]["q"]
    _close(r["mrr@10"], 0.5)
    _close(r["ndcg@10"], L3)  # 0.6309
    _close(r["recall@100"], 1.0)


def test_graded_gain():
    # DCG = 1/1 + 2/log2(3); IDCG = 2/1 + 1/log2(3)
    r = evaluate_ranked({"q": ["b", "a"]}, {"q": {"a": 2, "b": 1}})["per_query"]["q"]
    _close(r["ndcg@10"], (1 + 2 * L3) / (2 + L3))  # 0.85972
    _close(r["mrr@10"], 1.0)


def test_cutoffs():
    ranked = {"q": [f"junk{i}" for i in range(10)] + ["x"]}  # relevant doc at rank 11
    r = evaluate_ranked(ranked, {"q": {"x": 1}})["per_query"]["q"]
    _close(r["mrr@10"], 0.0)
    _close(r["ndcg@10"], 0.0)
    _close(r["recall@100"], 1.0)


def test_our_order_survives_ties():
    qrels = {"q": {"z": 1}}
    # Naive equal scores: trec_eval breaks ties by doc_id descending, putting "z" first.
    naive = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"}).evaluate({"q": {"a": 1.0, "z": 1.0}})
    _close(naive["q"]["recip_rank"], 1.0)
    # Our wrapper keeps the given order: "z" is rank 2.
    _close(evaluate_ranked({"q": ["a", "z"]}, qrels)["per_query"]["q"]["mrr@10"], 0.5)


def test_zero_relevance_not_relevant():
    r = evaluate_ranked({"q": ["n", "p"]}, {"q": {"n": 0, "p": 1}})["per_query"]["q"]
    _close(r["mrr@10"], 0.5)


def test_missing_query_raises():
    try:
        evaluate_ranked({"q1": ["a"]}, {"q1": {"a": 1}, "q2": {"b": 1}})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_mean_over_queries():
    out = evaluate_ranked({"q1": ["a"], "q2": ["x", "b"]}, {"q1": {"a": 1}, "q2": {"b": 1}})
    _close(out["mean"]["mrr@10"], 0.75)
    assert out["num_queries"] == 2


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")