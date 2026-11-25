# scripts/02b_tokenize_docfirst.py
"""Phase 3.1: doc-first variant B ("seam") tokenization.

Variant A (Phase 2): prefix + df_head + doc + "\\n<Query>:" + query + suffix
Variant B (here):    prefix + df_head + seam_doc + "<Query>:" + query + suffix
seam_doc = doc ids with the trailing newline baked in, so "." + "\\n" becomes the natural ".\\n".

Reads the Phase 2 artifact (never modifies it). Outputs:
  artifacts/tokenized/{ds}_qwen3_df.pt     seam doc ids, seam pieces, cache lengths, base artifact sha
  results/tokenization_docfirst.json       seam kinds, lengths, natural parity for A and B (embeds run_info)
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # must be set before transformers is imported

import argparse
import hashlib
import json
from collections import Counter

import numpy as np

from marginkv.config import load_config
from marginkv.data.beir_loader import load_beir
from marginkv.models.qwen3_reranker import (
    DEFAULT_INSTRUCTION,
    SEAM,
    build_pieces,
    check_tokenizer,
    doc_first_ids,
    doc_first_seam_ids,
    doc_piece_text,
    encode,
    load_tokenizer,
    natural_doc_first_ids,
    seam_doc_ids,
    seam_piece_texts,
)
from marginkv.retrieval.bm25 import doc_text
from marginkv.utils.io import load_pt, run_info, save_json, save_pt
from marginkv.utils.seed import set_seed

KINDS = ("merged", "appended", "fallback")


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
    ap.add_argument("--force", action="store_true", help="overwrite saved seam tokenization even if it differs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    max_doc = cfg.rerank.max_doc_tokens
    results, artifacts = cfg.paths.results_dir, cfg.paths.artifacts_dir
    instruction = DEFAULT_INSTRUCTION
    info = run_info()

    tok = load_tokenizer(cfg.models.qwen3.name, cfg.models.qwen3.revision)
    pieces = build_pieces(tok, instruction)
    check_tokenizer(tok, pieces)

    seam_texts = seam_piece_texts()
    seam_pieces = dict(zip(seam_texts, encode(tok, list(seam_texts.values()))))
    nl = encode(tok, [SEAM])[0]
    # Variant B's tag must be variant A's tag minus its leading newline token, so A and B
    # differ only at the seam. If this fails, paste the output: the design assumption is wrong.
    if len(nl) != 1 or pieces["df_mid"] != nl + seam_pieces["df_mid_seam"]:
        raise SystemExit(f"seam pieces don't line up: df_mid={tok.convert_ids_to_tokens(pieces['df_mid'])} "
                         f"nl={nl} df_mid_seam={tok.convert_ids_to_tokens(seam_pieces['df_mid_seam'])}")
    pieces_b = {**pieces, **seam_pieces}
    n_fixed = len(pieces["prefix"]) + len(pieces["df_head"])

    stats_out, summary = {}, []
    for name in cfg.datasets:
        base_path = artifacts / "tokenized" / f"{name}_qwen3.pt"
        art = load_pt(base_path)
        if art["pieces"] != pieces or art["instruction"] != instruction or art["max_doc_tokens"] != max_doc:
            raise SystemExit(f"[{name}] Phase 2 artifact doesn't match current pieces/instruction/max_doc_tokens")

        ds = load_beir(name, split=cfg.split, data_dir=cfg.paths.data_dir)
        doc_order, full_len, docs = art["doc_order"], art["doc_full_len"], art["doc"]
        untrunc = [d for d in doc_order if full_len[d] <= max_doc]
        is_trunc = {d: full_len[d] > max_doc for d in doc_order}

        # Re-tokenizing untruncated docs must reproduce the Phase 2 ids exactly (same text source).
        texts = {d: doc_text(ds.corpus[d]) for d in untrunc}
        re_ids = encode(tok, [doc_piece_text(texts[d]) for d in untrunc])
        drift = [d for d, r in zip(untrunc, re_ids) if r != docs[d]]
        if drift:
            raise SystemExit(f"[{name}] doc ids differ from the Phase 2 artifact for {len(drift)} docs: {drift[:5]}")

        doc_seam, kind = {}, {}
        for d in doc_order:
            doc_seam[d], kind[d] = seam_doc_ids(tok, docs[d], texts.get(d))  # None -> truncated path

        leaks = [d for d, ids in doc_seam.items() if any(i >= tok.vocab_size for i in ids)]
        if leaks:
            raise SystemExit(f"[{name}] special-token ids inside seam doc pieces: {leaks[:10]}")

        # Natural parity on pairs with untruncated docs: A should reproduce Phase 2 (mostly mismatches),
        # B should match everywhere.
        top, q_df = art["rerank_candidates"], art["query_df"]
        pairs = [(q, d) for q in art["query_order"] for d in top[q]]
        nat_pairs = [(q, d) for q, d in pairs if not is_trunc[d]]
        ref = natural_doc_first_ids(tok, instruction, [ds.queries[q] for q, _ in nat_pairs],
                                    [texts[d] for _, d in nat_pairs])
        match = {"A": 0, "B": 0}
        bad = []
        for (q, d), r in zip(nat_pairs, ref):
            a = doc_first_ids(pieces, docs[d], q_df[q])
            b = doc_first_seam_ids(pieces_b, doc_seam[d], q_df[q])
            match["A"] += a == r
            match["B"] += b == r
            if b != r and len(bad) < 5:
                bad.append(_diff_example(tok, q, d, b, r))

        lens_a = [len(doc_first_ids(pieces, docs[d], q_df[q])) for q, d in pairs]
        lens_b = [len(doc_first_seam_ids(pieces_b, doc_seam[d], q_df[q])) for q, d in pairs]
        cache_len = {d: n_fixed + len(doc_seam[d]) for d in doc_order}

        def _kinds(ds_):
            c = Counter(kind[d] for d in ds_)
            return {k: c.get(k, 0) for k in KINDS}

        st = {
            "num_pairs": len(pairs),
            "num_unique_docs": len(doc_order),
            "seam_kinds": {"untruncated": _kinds(untrunc),
                           "truncated": _kinds([d for d in doc_order if is_trunc[d]])},
            "doc_seam_tokens": _len_stats([len(v) for v in doc_seam.values()]),
            "cache_len_seam": _len_stats(list(cache_len.values())),
            "prompt_tokens_doc_first_A": _len_stats(lens_a),
            "prompt_tokens_doc_first_B": _len_stats(lens_b),
            "natural_parity": {"checked": len(nat_pairs), "matched_A": match["A"], "matched_B": match["B"],
                               "B_mismatch_examples": bad},
        }

        if match["B"] != len(nat_pairs):
            print(json.dumps(st["natural_parity"], indent=2, ensure_ascii=False))
            print(json.dumps(st["seam_kinds"], indent=2))
            raise SystemExit(f"[{name}] variant B natural parity failed. Paste this output. Nothing saved.")

        content = {
            "dataset": name,
            "split": cfg.split,
            "model": {"name": cfg.models.qwen3.name, "revision": cfg.models.qwen3.revision},
            "instruction": instruction,
            "variant": "seam",
            "base_artifact": str(base_path.relative_to(cfg.paths.artifacts_dir.parent)),
            "base_content_sha256": art["content_sha256"],
            "max_doc_tokens": max_doc,
            "seam_piece_texts": seam_texts,
            "seam_pieces": seam_pieces,
            "doc_order": doc_order,
            "doc_seam": doc_seam,          # {doc_id: ids}, <= max_doc + 1
            "seam_kind": kind,             # {doc_id: merged | appended | fallback}
            "cache_len_seam": cache_len,   # {doc_id: len(prefix + df_head + doc_seam)} = query start position
        }
        sha = _sha(content)
        st["content_sha256"] = sha

        path = artifacts / "tokenized" / f"{name}_qwen3_df.pt"
        if path.exists() and not args.force:
            old = load_pt(path)
            if old.get("content_sha256") != sha:
                raise SystemExit(f"[{name}] new seam tokenization differs from {path}. Investigate, or rerun with --force.")
            print(f"[{name}] seam tokenization unchanged: {path}")
        else:
            save_pt({**content, "content_sha256": sha, "run_info": info}, path)
            print(f"[{name}] saved {path}")
        st["artifact"] = str(path.relative_to(cfg.paths.artifacts_dir.parent))
        stats_out[name] = st
        summary.append((name, st))

    save_json({
        "run_info": info,
        "model": {"name": cfg.models.qwen3.name, "revision": cfg.models.qwen3.revision},
        "instruction": instruction,
        "seam_piece_lengths": {k: len(v) for k, v in seam_pieces.items()},
        "datasets": stats_out,
    }, results / "tokenization_docfirst.json")

    print(f"\n{'dataset':10s} {'natA':>10s} {'natB':>10s} {'untr m/a/f':>14s} {'trunc m/a/f':>16s} "
          f"{'seamMax':>7s} {'dfA max':>7s} {'dfB max':>7s}")
    for name, st in summary:
        p, k = st["natural_parity"], st["seam_kinds"]
        u, t = k["untruncated"], k["truncated"]
        print(f"{name:10s} {p['matched_A']:>4d}/{p['checked']:<5d} {p['matched_B']:>4d}/{p['checked']:<5d} "
              f"{u['merged']:>4d}/{u['appended']}/{u['fallback']:<4d} "
              f"{t['merged']:>5d}/{t['appended']}/{t['fallback']:<5d} "
              f"{st['doc_seam_tokens']['max']:>7d} {st['prompt_tokens_doc_first_A']['max']:>7d} "
              f"{st['prompt_tokens_doc_first_B']['max']:>7d}")


if __name__ == "__main__":
    main()