# src/marginkv/retrieval/bm25.py
"""BM25 first stage (bm25s): index title+text, score full corpus, deterministic top-k."""
from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass

import bm25s
import numpy as np
import Stemmer

from marginkv.data.beir_loader import BeirDataset, load_beir


@dataclass(frozen=True)
class BM25Params:
    method: str = "lucene"
    k1: float = 1.2
    b: float = 0.75
    stopwords: str = "en"
    stemmer: str = "english"
    lower: bool = True


def doc_text(doc: dict[str, str]) -> str:
    return f"{doc['title']} {doc['text']}".strip()


class BM25Index:
    def __init__(self, corpus: dict[str, dict[str, str]], params: BM25Params = BM25Params()):
        self.params = params
        self.doc_ids = list(corpus.keys())
        self._stemmer = Stemmer.Stemmer(params.stemmer)

        tokens = self._tokenize([doc_text(corpus[d]) for d in self.doc_ids])
        self.retriever = bm25s.BM25(method=params.method, k1=params.k1, b=params.b)
        self.retriever.index(tokens, show_progress=False)
        self.vocab = self.retriever.vocab_dict  # token -> id

        # Tie-break key: each doc's position in lexicographic doc_id order.
        order = sorted(range(len(self.doc_ids)), key=lambda i: self.doc_ids[i])
        self._id_rank = np.empty(len(self.doc_ids), dtype=np.int64)
        self._id_rank[order] = np.arange(len(self.doc_ids))

    def _tokenize(self, texts: list[str]) -> list[list[str]]:
        out = bm25s.tokenize(
            texts, lower=self.params.lower, stopwords=self.params.stopwords,
            stemmer=self._stemmer, return_ids=False, show_progress=False,
        )
        # Defensive: some bm25s versions return a Tokenized(ids, vocab) tuple.
        if hasattr(out, "ids") and hasattr(out, "vocab"):
            inv = {v: k for k, v in out.vocab.items()}
            return [[inv[i] for i in ids] for ids in out.ids]
        return [list(t) for t in out]

    def query_terms(self, query: str) -> list[str]:
        return [t for t in self._tokenize([query])[0] if t in self.vocab]

    def scores(self, query: str) -> np.ndarray:
        terms = self.query_terms(query)
        if not terms:
            return np.zeros(len(self.doc_ids), dtype=np.float64)
        return np.asarray(self.retriever.get_scores(terms), dtype=np.float64)

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        s = self.scores(query)
        order = np.lexsort((self._id_rank, -s))[:k]  # primary: -score, secondary: doc_id
        return [(self.doc_ids[i], float(s[i])) for i in order]


def retrieve_all(ds: BeirDataset, k: int = 100, params: BM25Params = BM25Params()):
    """Returns (run, info). run: {qid: [(doc_id, score), ...]} in rank order, query file order."""
    t0 = time.perf_counter()
    index = BM25Index(ds.corpus, params)
    run: dict[str, list[tuple[str, float]]] = {}
    empty_terms, short_pool = [], []
    for qid, query in ds.queries.items():
        hits = index.search(query, k)
        ids = [d for d, _ in hits]
        if len(ids) != k or len(set(ids)) != k:
            raise RuntimeError(f"[{ds.name}] query {qid}: expected {k} unique docs, got {len(set(ids))}")
        if not index.query_terms(query):
            empty_terms.append(qid)
        if sum(1 for _, s in hits if s > 0) < k:
            short_pool.append(qid)
        run[qid] = hits
    info = {
        "dataset": ds.name,
        "bm25": asdict(params),
        "bm25s_version": getattr(bm25s, "__version__", "unknown"),
        "doc_text": "title + ' ' + text",
        "k": k,
        "num_docs": len(ds.corpus),
        "num_queries": len(run),
        "vocab_size": len(index.vocab),
        "queries_with_no_known_terms": empty_terms,
        "queries_with_fewer_than_k_matches": len(short_pool),
        "tie_break": "score desc, then doc_id asc",
        "seconds": round(time.perf_counter() - t0, 2),
    }
    return run, info


def _quick_recall(run, qrels) -> float:
    """Hand-rolled Recall@k, cross-checked against pytrec_eval in 1.3."""
    vals = []
    for qid, hits in run.items():
        pos = {d for d, r in qrels[qid].items() if r > 0}
        got = {d for d, _ in hits}
        vals.append(len(pos & got) / len(pos))
    return sum(vals) / len(vals)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["scifact", "nfcorpus"]:
        ds = load_beir(name, download=False)
        run1, info = retrieve_all(ds)
        run2, _ = retrieve_all(ds)
        print({k: v for k, v in info.items() if k != "bm25"})
        print(f"  deterministic: {run1 == run2}")
        print(f"  quick Recall@100: {_quick_recall(run1, ds.qrels):.4f}")
        qid = next(iter(run1))
        print(f"  first query {qid}: {ds.queries[qid][:80]!r}")
        for rank, (d, s) in enumerate(run1[qid][:3], 1):
            rel = ds.qrels[qid].get(d, 0)
            print(f"    {rank}. {d}  score={s:.3f}  rel={rel}  {ds.corpus[d]['title'][:60]!r}")