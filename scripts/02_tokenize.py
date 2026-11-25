# scripts/02_tokenize.py
"""Phase 2.1: tokenize queries and top-k candidate docs once for Qwen3-Reranker.

Outputs:
  artifacts/tokenized/{ds}_qwen3.pt   prompt pieces, query ids (both orders), truncated doc ids
  results/tokenization.json           lengths, truncation, tokenization parity checks (embeds run_info)

Every later Qwen3 experiment builds prompts from the .pt file and never re-tokenizes.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # must be set before transformers is imported

import argparse
import hashlib
import json

import numpy as np

from marginkv.config import load_config
from marginkv.data.beir_loader import load_beir
from marginkv.models.qwen3_reranker import (
    DEFAULT_INSTRUCTION,
    build_pieces,
    check_tokenizer,
    doc_first_ids,
    doc_piece_text,
    encode,
    load_tokenizer,
    natural_doc_first_ids,
    official_query_first_ids,
    piece_texts,
    query_first_ids,
    query_text_df,
    query_text_qf,
)
from marginkv.retrieval.bm25 import doc_text
from marginkv.utils.io import load_json, load_pt, run_info, save_json, save_pt
from marginkv.utils.seed import set_seed


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _len_stats(xs) -> dict:
    a = np.asarray(xs)
    return {"min": int(a.min()), "mean": round(float(a.mean()), 2), "p50": int(np.percentile(a, 50)),
            "p95": int(np.percentile(a, 95)), "max": int(a.max())}


def _diff_example(tok, qid, did, ours, ref) -> dict:
    i = next((k for k, (x, y) in enumerate(zip(ours, ref)) if x != y), min(len(ours), len(ref)))
    lo = max(0, i - 3)
    return {"qid": qid, "doc_id": did, "pos": i,
            "ours": tok.convert_ids_to_tokens(ours[lo:i + 3]),
            "ref": tok.convert_ids_to_tokens(ref[lo:i + 3])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="overwrite saved tokenization even if it differs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    depth, max_doc = cfg.rerank.depth, cfg.rerank.max_doc_tokens
    results, artifacts = cfg.paths.results_dir, cfg.paths.artifacts_dir
    instruction = DEFAULT_INSTRUCTION
    info = run_info()

    tok = load_tokenizer(cfg.models.qwen3.name, cfg.models.qwen3.revision)
    pieces = build_pieces(tok, instruction)
    check_tokenizer(tok, pieces)
    n_suf = len(pieces["suffix"])

    stats_out, summary = {}, []
    for name in cfg.datasets:
        ds = load_beir(name, split=cfg.split, data_dir=cfg.paths.data_dir)
        cand = load_json(results / "candidates" / f"{name}.json")
        query_order = cand["query_order"]
        if query_order != list(ds.queries):
            raise SystemExit(f"[{name}] candidate query_order differs from the dataset's query order")

        top = {}
        for qid in query_order:
            hits = cand["candidates"][qid]
            if len(hits) < depth:
                raise SystemExit(f"[{name}] query {qid} has {len(hits)} candidates < depth {depth}")
            top[qid] = [d for d, _ in hits[:depth]]
        doc_order = sorted({d for docs in top.values() for d in docs})

        # Tokenize every variable piece once.
        q_texts = [ds.queries[q] for q in query_order]
        q_qf = dict(zip(query_order, encode(tok, [query_text_qf(t) for t in q_texts])))
        q_df = dict(zip(query_order, encode(tok, [query_text_df(t) for t in q_texts])))
        d_texts = {d: doc_text(ds.corpus[d]) for d in doc_order}
        full = encode(tok, [doc_piece_text(d_texts[d]) for d in doc_order])
        doc_full_len = {d: len(f) for d, f in zip(doc_order, full)}
        doc_ids = {d: f[:max_doc] for d, f in zip(doc_order, full)}  # truncate after tokenizing

        # A literal "<|im_end|>" etc. inside text would be parsed as a control token. Refuse.
        leaks = [k for k, ids in [*q_qf.items(), *q_df.items(), *zip(doc_order, full)]
                 if any(i >= tok.vocab_size for i in ids)]
        if leaks:
            raise SystemExit(f"[{name}] special-token ids inside text pieces: {leaks[:10]}")

        # Parity check: our query-first concat vs the model card's tokenization, on every reranked pair.
        # Untruncated doc: full equality. Truncated doc: ours (minus suffix) must be a prefix of the
        # official ids built from the full doc, i.e. we cut exactly at a token the model card would produce.
        pairs = [(q, d) for q in query_order for d in top[q]]
        ref = official_query_first_ids(tok, instruction, [ds.queries[q] for q, _ in pairs],
                                       [d_texts[d] for _, d in pairs])
        chk = {"full": [0, 0], "truncated": [0, 0]}  # [checked, matched]
        bad, qf_lens, df_lens = [], [], []
        for (q, d), r in zip(pairs, ref):
            ours = query_first_ids(pieces, q_qf[q], doc_ids[d])
            qf_lens.append(len(ours))
            df_lens.append(len(doc_first_ids(pieces, doc_ids[d], q_df[q])))
            if doc_full_len[d] <= max_doc:
                kind, ok = "full", ours == r
            else:
                body = ours[:-n_suf]
                kind, ok = "truncated", r[:len(body)] == body
            chk[kind][0] += 1
            chk[kind][1] += ok
            if not ok and len(bad) < 5:
                bad.append(_diff_example(tok, q, d, ours, r))

        # Diagnostic: doc-first concat vs tokenizing the doc-first string in one call (untruncated only).
        # Mismatches here are expected when a doc ends in punctuation (".\n" merge). Recorded, not fixed.
        df_pairs = [(q, d) for q, d in pairs if doc_full_len[d] <= max_doc]
        df_ref = natural_doc_first_ids(tok, instruction, [ds.queries[q] for q, _ in df_pairs],
                                       [d_texts[d] for _, d in df_pairs])
        df_match, df_bad = 0, []
        for (q, d), r in zip(df_pairs, df_ref):
            ours = doc_first_ids(pieces, doc_ids[d], q_df[q])
            if ours == r:
                df_match += 1
            elif len(df_bad) < 3:
                df_bad.append(_diff_example(tok, q, d, ours, r))

        n_trunc = sum(doc_full_len[d] > max_doc for d in doc_order)
        st = {
            "num_queries": len(query_order),
            "num_pairs": len(pairs),
            "num_unique_docs": len(doc_order),
            "doc_tokens_before_truncation": _len_stats(list(doc_full_len.values())),
            "docs_truncated": n_trunc,
            "docs_truncated_frac": round(n_trunc / len(doc_order), 4),
            "query_tokens_qf": _len_stats([len(v) for v in q_qf.values()]),
            "prompt_tokens_query_first": _len_stats(qf_lens),
            "prompt_tokens_doc_first": _len_stats(df_lens),
            "qf_parity": {k: {"checked": c, "matched": m} for k, (c, m) in chk.items()},
            "qf_parity_mismatch_examples": bad,
            "df_natural_parity": {"checked": len(df_pairs), "matched": df_match, "examples": df_bad},
        }

        ok_all = all(c == m for c, m in chk.values())
        if not ok_all:
            print(json.dumps(st["qf_parity"], indent=2))
            print(json.dumps(bad, indent=2, ensure_ascii=False))
            raise SystemExit(f"[{name}] query-first parity failed. Paste this output. Nothing saved.")

        content = {
            "dataset": name,
            "split": cfg.split,
            "model": {"name": cfg.models.qwen3.name, "revision": cfg.models.qwen3.revision},
            "instruction": instruction,
            "max_doc_tokens": max_doc,
            "depth": depth,
            "piece_texts": piece_texts(instruction),
            "pieces": pieces,
            "query_order": query_order,
            "query_qf": q_qf,
            "query_df": q_df,
            "doc_order": doc_order,
            "doc": doc_ids,
            "doc_full_len": doc_full_len,
            "rerank_candidates": top,  # {qid: [doc_id x depth]}, BM25 rank order
        }
        sha = _sha(content)
        st["content_sha256"] = sha

        path = artifacts / "tokenized" / f"{name}_qwen3.pt"
        if path.exists() and not args.force:
            old = load_pt(path)
            if old.get("content_sha256") != sha:
                raise SystemExit(f"[{name}] new tokenization differs from {path}. Investigate, or rerun with --force.")
            print(f"[{name}] tokenization unchanged: {path}")
        else:
            save_pt({**content, "content_sha256": sha, "run_info": info}, path)
            print(f"[{name}] saved {path}")
        st["artifact"] = str(path.relative_to(cfg.paths.artifacts_dir.parent))
        stats_out[name] = st
        summary.append((name, st))

    save_json({
        "run_info": info,
        "model": {"name": cfg.models.qwen3.name, "revision": cfg.models.qwen3.revision},
        "tokenizer": {"class": type(tok).__name__, "vocab_size": tok.vocab_size},
        "instruction": instruction,
        "piece_lengths": {k: len(v) for k, v in pieces.items()},
        "datasets": stats_out,
    }, results / "tokenization.json")

    print(f"\npieces: {{{', '.join(f'{k}: {len(v)}' for k, v in pieces.items())}}}")
    print(f"{'dataset':10s} {'docs':>5s} {'trunc':>6s} {'docTok p50':>10s} {'qfMax':>6s} "
          f"{'dfMax':>6s} {'qfParity':>9s} {'dfNatural':>10s}")
    for name, st in summary:
        p = st["qf_parity"]
        n_ok = p["full"]["matched"] + p["truncated"]["matched"]
        n = p["full"]["checked"] + p["truncated"]["checked"]
        dfp = st["df_natural_parity"]
        print(f"{name:10s} {st['num_unique_docs']:5d} {st['docs_truncated_frac']:6.1%} "
              f"{st['doc_tokens_before_truncation']['p50']:10d} {st['prompt_tokens_query_first']['max']:6d} "
              f"{st['prompt_tokens_doc_first']['max']:6d} {n_ok:>4d}/{n:<4d} "
              f"{dfp['matched']:>4d}/{dfp['checked']:<5d}")


if __name__ == "__main__":
    main()