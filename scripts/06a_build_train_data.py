# scripts/06a_build_train_data.py
"""Phase 4.1: build MS MARCO train/dev groups, tokenized for both prompt orders.

Output: artifacts/train/msmarco_qwen3.pt (+ results/train_data.json).
Smoke: --smoke N writes under artifacts/smoke/ only.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import random
import sys
import time

from huggingface_hub import hf_hub_download

from marginkv.config import load_config
from marginkv.data.beir_loader import load_beir
from marginkv.models import qwen3_reranker as qr
from marginkv.train import data as td
from marginkv.utils.io import ensure_dir, load_json, load_pt, run_info, save_json, save_pt
from marginkv.utils.seed import set_seed


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def len_stats(xs):
    return {"n": len(xs), "mean": round(sum(xs) / len(xs), 2), "p50": pct(xs, 50),
            "p95": pct(xs, 95), "max": max(xs)}


def get_file(cfg):
    if td.TEVATRON_REVISION == "PASTE_SHA_HERE":
        sys.exit("Pin td.TEVATRON_REVISION first (see run instructions).")
    ddir = ensure_dir(cfg.paths.data_dir / "msmarco_tevatron")
    path = ddir / td.TEVATRON_FILE
    man_path = ddir / "manifest.json"
    if not path.exists():
        hf_hub_download(repo_id=td.TEVATRON_REPO, repo_type="dataset", filename=td.TEVATRON_FILE,
                        revision=td.TEVATRON_REVISION, local_dir=ddir)
    size = path.stat().st_size
    man = load_json(man_path) if man_path.exists() else None
    if man and man["revision"] == td.TEVATRON_REVISION and man["size"] == size:
        sha = man["sha256"]
    else:
        print("hashing train file ...", flush=True)
        sha = td.sha256_file(path)
        save_json({"repo": td.TEVATRON_REPO, "revision": td.TEVATRON_REVISION,
                   "file": td.TEVATRON_FILE, "size": size, "sha256": sha}, man_path)
    return path, {"repo": td.TEVATRON_REPO, "revision": td.TEVATRON_REVISION,
                  "file": td.TEVATRON_FILE, "size": size, "sha256": sha}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-dev", type=int, default=1000)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--neg-pool", type=int, default=15)
    ap.add_argument("--parity-queries", type=int, default=3000)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)
    t0 = time.time()
    n_train, n_dev = args.n_train, args.n_dev
    if args.smoke:
        n_train, n_dev = args.smoke, max(1, args.smoke // 4)
    base = cfg.paths.artifacts_dir / "smoke" if args.smoke else cfg.paths.artifacts_dir
    out_pt = base / "train" / "msmarco_qwen3.pt"
    out_json = (base / "train_data.json") if args.smoke else (cfg.paths.results_dir / "train_data.json")
    max_doc = cfg.rerank.max_doc_tokens

    # 1. source file
    path, source = get_file(cfg)

    # 2. tokenizer + pieces, asserted equal to the frozen Phase 2 / 3.1 artifacts
    tok = qr.load_tokenizer(cfg.models.qwen3.name, cfg.models.qwen3.revision)
    inst = qr.DEFAULT_INSTRUCTION
    pieces = qr.build_pieces(tok, inst)
    qr.check_tokenizer(tok, pieces)
    seam = {k: qr.encode(tok, [v])[0] for k, v in qr.seam_piece_texts().items()}
    ref_ds = cfg.datasets[0]
    art = load_pt(cfg.paths.artifacts_dir / "tokenized" / f"{ref_ds}_qwen3.pt")
    dfart = load_pt(cfg.paths.artifacts_dir / "tokenized" / f"{ref_ds}_qwen3_df.pt")
    assert art["pieces"] == pieces, "pieces differ from Phase 2 artifact"
    assert dfart["seam_pieces"] == seam, "seam pieces differ from Phase 3.1 artifact"
    assert art["instruction"] == inst and art["max_doc_tokens"] == max_doc
    pieces = {**pieces, **seam}

    # 3. BEIR test texts for the leakage check
    test_q, test_d = set(), set()
    for ds in cfg.datasets:
        b = load_beir(ds, split=cfg.split, data_dir=cfg.paths.data_dir, download=False)
        test_q |= {td.normalize_text(q) for q in b.queries.values()}
        for d in b.corpus.values():
            test_d.add(td.normalize_text(d["text"]))
            test_d.add(td.normalize_text(f"{d['title']} {d['text']}"))
    print(f"test queries {len(test_q)}, test doc texts {len(test_d)}", flush=True)

    # 4. sample line indices (seeded), parse only those
    print("counting lines ...", flush=True)
    total = td.count_lines(path)
    rng = random.Random(cfg.seed)
    n_pick = min(total, int((n_train + n_dev) * 1.15) + 20)
    picked = rng.sample(range(total), n_pick)  # random order = assignment order
    print(f"rows {total}, parsing {n_pick} sampled rows ...", flush=True)
    rows = td.read_selected(path, set(picked))

    # 5. filter + build groups; dev filled first, then train
    drops = {"leak_query": 0, "leak_doc": 0, "dup_query": 0, "no_positive": 0, "few_negatives": 0}
    split = {"dev": [], "train": []}
    query_text, groups, texts, hard = {}, {}, {}, []
    seen_q = set()
    for idx in picked:
        if len(split["dev"]) >= n_dev and len(split["train"]) >= n_train:
            break
        row = rows[idx]
        qid, q = str(row["query_id"]), row["query"].strip()
        qn = td.normalize_text(q)
        if qn in test_q:
            drops["leak_query"] += 1
            continue
        if qn in seen_q or qid in groups:
            drops["dup_query"] += 1
            continue
        g, tx, h = td.select_group(row, args.group_size, args.neg_pool, cfg.seed)
        if g is None:
            drops[tx] += 1
            continue
        if any(td.normalize_text(t) in test_d for t in tx.values()):
            drops["leak_doc"] += 1
            continue
        which = "dev" if len(split["dev"]) < n_dev else "train"
        split[which].append(qid)
        seen_q.add(qn)
        query_text[qid] = q
        groups[qid] = g
        texts.update(tx)
        hard.append(h)
    if len(split["train"]) < n_train or len(split["dev"]) < n_dev:
        sys.exit(f"not enough rows: train {len(split['train'])}, dev {len(split['dev'])}")

    # 6. tokenize queries (both orders) and passages (plain + seam)
    qids = split["dev"] + split["train"]
    q_qf = dict(zip(qids, qr.encode(tok, [qr.query_text_qf(query_text[i]) for i in qids])))
    q_df = dict(zip(qids, qr.encode(tok, [qr.query_text_df(query_text[i]) for i in qids])))
    pids = sorted(texts)
    full = qr.encode(tok, [qr.doc_piece_text(texts[p]) for p in pids])
    doc, doc_seam, seam_kind, full_len = {}, {}, {}, {}
    for p, ids in zip(pids, full):
        full_len[p] = len(ids)
        doc[p] = ids[:max_doc]
        untrunc = len(ids) <= max_doc
        s, kind = qr.seam_doc_ids(tok, doc[p], text=texts[p] if untrunc else None)
        assert len(s) <= max_doc + 1, (p, len(s))
        doc_seam[p], seam_kind[p] = s, kind

    special = set(tok.all_special_ids)
    bad = [p for p in pids if special & set(doc_seam[p])] + [i for i in qids if special & set(q_qf[i])]
    if bad:
        sys.exit(f"special token ids inside text for {len(bad)} items, e.g. {bad[:5]}")

    # 7. parity on untruncated pairs: our pieces vs one-call tokenization.
    # Fallback docs (doc + "\n") differ from natural tokenization by design.
    par_q = split["dev"] + split["train"][: args.parity_queries]
    qs, ds_, kinds_, pids_, ours_qf, ours_df = [], [], [], [], [], []
    for qid in par_q:
        for p in groups[qid]:
            if full_len[p] <= max_doc:
                qs.append(query_text[qid])
                ds_.append(texts[p])
                kinds_.append(seam_kind[p])
                pids_.append(p)
                ours_qf.append(qr.query_first_ids(pieces, q_qf[qid], doc[p]))
                ours_df.append(qr.doc_first_seam_ids(pieces, doc_seam[p], q_df[qid]))
    ref_qf, ref_df = [], []
    for i in range(0, len(qs), 512):
        ref_qf += qr.official_query_first_ids(tok, inst, qs[i:i + 512], ds_[i:i + 512])
        ref_df += qr.natural_doc_first_ids(tok, inst, qs[i:i + 512], ds_[i:i + 512])
    qf_ok = sum(a == b for a, b in zip(ours_qf, ref_qf))
    df_ok = sum(a == b for a, b in zip(ours_df, ref_df))
    by_kind = {}
    bad_examples = []
    for k, p, a, b in zip(kinds_, pids_, ours_df, ref_df):
        e = by_kind.setdefault(k, {"pairs": 0, "match": 0})
        e["pairs"] += 1
        e["match"] += int(a == b)
        if a != b and len(bad_examples) < 5:
            bad_examples.append({"pid": p, "kind": k, "tail": texts[p][-40:]})
    parity = {"pairs": len(qs), "query_first_match": qf_ok, "doc_first_seam_match": df_ok,
              "doc_first_by_kind": by_kind, "mismatch_examples": bad_examples}

    # 8. stats
    seq_qf = [len(qr.query_first_ids(pieces, q_qf[q], doc[p])) for q in qids for p in groups[q]]
    seq_df = [len(qr.doc_first_seam_ids(pieces, doc_seam[p], q_df[q])) for q in qids for p in groups[q]]
    dev_p = {p for q in split["dev"] for p in groups[q]}
    train_p = {p for q in split["train"] for p in groups[q]}
    kinds = {}
    for k in seam_kind.values():
        kinds[k] = kinds.get(k, 0) + 1
    stats = {
        "rows_total": total, "rows_parsed": n_pick, "drops": drops,
        "num_train": len(split["train"]), "num_dev": len(split["dev"]),
        "num_passages": len(pids), "dev_passages_also_in_train": len(dev_p & train_p),
        "passage_tokens_full": len_stats(list(full_len.values())),
        "truncated_frac": round(sum(v > max_doc for v in full_len.values()) / len(pids), 4),
        "query_tokens_df": len_stats([len(q_df[q]) for q in qids]),
        "seq_len_qf": len_stats(seq_qf), "seq_len_df": len_stats(seq_df),
        "seam_kind": kinds,
        "hardness_mean": {k: round(sum(h[k] for h in hard) / len(hard), 4) for k in hard[0]},
        "parity": parity,
    }

    # 9. content + guard + save
    content = {
        "dataset": "msmarco-passage", "source": source,
        "model": {"name": cfg.models.qwen3.name, "revision": cfg.models.qwen3.revision},
        "instruction": inst, "max_doc_tokens": max_doc, "group_size": args.group_size,
        "neg_pool": args.neg_pool, "seed": cfg.seed, "pieces": pieces, "split": split,
        "query_text": query_text, "query_qf": q_qf, "query_df": q_df, "groups": groups,
        "doc": doc, "doc_seam": doc_seam, "seam_kind": seam_kind, "doc_full_len": full_len,
    }
    sha = td.content_sha256(content)
    if out_pt.exists() and not args.force:
        old = load_pt(out_pt, weights_only=False)
        if old.get("content_sha256") != sha:
            sys.exit(f"{out_pt} exists with different content; use --force")
        print("existing artifact has identical content", flush=True)
    ensure_dir(out_pt.parent)
    save_pt({**content, "content_sha256": sha, "run_info": run_info()}, out_pt)
    save_json({"run_info": run_info(), "source": source, "content_sha256": sha,
               "settings": vars(args), "stats": stats}, out_json)

    print(f"train {stats['num_train']}  dev {stats['num_dev']}  passages {stats['num_passages']}")
    print(f"drops {drops}")
    print(f"passage tokens {stats['passage_tokens_full']}  truncated {stats['truncated_frac']}")
    print(f"seq qf {stats['seq_len_qf']}\nseq df {stats['seq_len_df']}")
    print(f"seam kinds {kinds}")
    print(f"hardness {stats['hardness_mean']}")
    non_fb = [v for k, v in by_kind.items() if k != "fallback"]
    df_clean = all(v["match"] == v["pairs"] for v in non_fb)
    status = "OK" if qf_ok == len(qs) and df_clean else "PARITY FAIL"
    print(f"parity qf {qf_ok}/{len(qs)}  df by kind {by_kind}  {status}")
    for ex in bad_examples:
        print(f"  mismatch {ex['kind']} {ex['pid']}: ...{ex['tail']!r}")
    print(f"sha {sha}  ({time.time() - t0:.0f} s)  -> {out_pt}")


if __name__ == "__main__":
    main()