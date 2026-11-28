# scripts/06e_build_train_data_long.py
"""Phase 8b.2: ablation B training data from MS MARCO v1 document ranking (official files).

FirstP: doc text = title + " " + body, cut to CHAR_CUT chars, window = first 256 Qwen tokens (same truncation as
BEIR). Option B: keep a query only if its positive passage (Tevatron, same qid) appears inside the window. Groups
use the Phase 4.1 rules (train.data.select_group) on the window text, negatives from the official BM25 top 100
(ranks 1..30). Leakage: exact test queries, 13-word shingles of every SciFact/NFCorpus test doc, nutritionfacts.org
URLs. Output keys equal 06a's, so 06_train_lora can read the artifact.

Output: artifacts/train/msmarco_doc_qwen3.pt (+ results/train_data_long.json). Doc-file passes cached under
artifacts/train/msmarco_doc_cache/.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from array import array

from marginkv.config import load_config
from marginkv.data.beir_loader import load_beir
from marginkv.models import qwen3_reranker as qr
from marginkv.train import data as td
from marginkv.utils.io import ensure_dir, load_json, load_pt, run_info, save_json, save_pt
from marginkv.utils.seed import set_seed

BASE_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/"
FILES = {"queries": "msmarco-doctrain-queries.tsv.gz", "qrels": "msmarco-doctrain-qrels.tsv.gz",
         "top100": "msmarco-doctrain-top100.gz", "docs": "msmarco-docs.tsv.gz"}
CHAR_CUT = 3000
MIN_CUT_TOKENS = 300
NEG_DEPTH = 30
SHINGLE = 13
LEAK_DOMAINS = ("nutritionfacts.org",)
_ALNUM = re.compile(r"[a-z0-9]+")
_TEV_QID = re.compile(r'^\{"query_id":\s*"([^"]+)"')

RULES = {
    "phase": "8b.2",
    "source": "official MS MARCO v1 document ranking train files (FILES), sha256 recorded per file",
    "doc_text": "title + ' ' + body; title dropped if empty, '-' or '.'",
    "char_cut": f"text cut to the first {CHAR_CUT} chars before tokenizing; a cut doc must have >= {MIN_CUT_TOKENS} "
                "tokens, else it is unusable (a negative is skipped; a positive drops its query). doc_full_len of a "
                "cut doc = tokens of the cut text (a lower bound)",
    "window": "first max_doc_tokens (256) Qwen tokens of ' ' + text (FirstP), same truncation as BEIR",
    "pool": "doctrain qids with a qrel and a query, sorted, shuffled with random.Random(f'{seed}-msmarco-doc'), first `pool`",
    "answer_in_window": "option B: some Tevatron positive passage text of the same qid, as an ASCII lowercase "
                        "alphanumeric word sequence, appears contiguously in the window's word sequence; qids absent "
                        "from Tevatron are dropped",
    "negatives": f"official doctrain top100 ranks 1..{NEG_DEPTH} in rank order; train.data.select_group on the window "
                 "text (dedup by id and normalized text, never equal to the positive, top neg_pool by stemmed "
                 "overlap, seeded sample of group_size - 1)",
    "leakage": f"drop a query whose normalized text equals a test query; drop a group if any doc's cut text contains a "
               f"{SHINGLE}-word shingle of any SciFact/NFCorpus test doc (title + text) or its URL contains {LEAK_DOMAINS}",
    "split": "dev filled first, then train, in pool order",
    "tokenization": "same pieces as Phase 2 / 3.1 (asserted); doc = window ids; seam variant B; queries in both orders",
}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else None


def len_stats(xs):
    xs = list(xs)
    return {"n": len(xs), "mean": round(sum(xs) / len(xs), 2), "p50": pct(xs, 50), "p95": pct(xs, 95), "max": max(xs)}


def gz_lines(path):
    """Raw byte lines via the system gzip (faster than Python's); kills it if the reader stops early."""
    p = subprocess.Popen(["gzip", "-dc", str(path)], stdout=subprocess.PIPE, bufsize=1 << 20)
    done = False
    try:
        for line in p.stdout:
            yield line
        done = True
    finally:
        p.stdout.close()
        if not done:
            p.kill()
        rc = p.wait()
    if done and rc != 0:
        raise RuntimeError(f"gzip -dc failed on {path} (rc {rc})")


def words(s):
    return _ALNUM.findall(s.lower())


def doc_text(title, body):
    title, body = title.strip(), body.strip()
    return f"{title} {body}" if title and title not in ("-", ".") else body


def sources(cfg):
    ddir = cfg.paths.data_dir / "msmarco_doc"
    man_path = ddir / "manifest.json"
    man = load_json(man_path) if man_path.exists() else {}
    out = {}
    for key, name in FILES.items():
        path = ddir / name
        if not path.exists():
            sys.exit(f"missing {path}; download it first")
        size = path.stat().st_size
        old = man.get(name)
        if old and old["size"] == size:
            sha = old["sha256"]
        else:
            print(f"hashing {name} ...", flush=True)
            sha = td.sha256_file(path)
        out[key] = {"file": name, "url": BASE_URL + name, "size": size, "sha256": sha, "path": path}
    save_json({v["file"]: {k: v[k] for k in ("url", "size", "sha256")} for v in out.values()}, man_path)
    tdir = cfg.paths.data_dir / "msmarco_tevatron"
    tman = load_json(tdir / "manifest.json")
    tsize = (tdir / td.TEVATRON_FILE).stat().st_size
    if tman["revision"] != td.TEVATRON_REVISION or tman["size"] != tsize:
        sys.exit("Tevatron file differs from its Phase 4 manifest")
    out["tevatron"] = {**tman, "path": tdir / td.TEVATRON_FILE}
    return out


def read_queries(path):
    out = {}
    for line in gz_lines(path):
        qid, _, q = line.decode("utf-8", "replace").rstrip("\n").partition("\t")
        if qid and q.strip():
            out[qid] = q.strip()
    return out


def read_qrels(path):
    out = {}
    for line in gz_lines(path):
        f = line.decode().split()
        if len(f) >= 4 and int(f[3]) > 0:
            out.setdefault(f[0], set()).add(f[2])
    return {q: sorted(v) for q, v in out.items()}


def read_top(path, qids, depth):
    want, out = {q.encode() for q in qids}, {}
    for line in gz_lines(path):
        q = line[:line.find(b" ")]
        if q not in want:
            continue
        f = line.decode().split()
        if int(f[3]) <= depth:
            out.setdefault(f[0], []).append((int(f[3]), f[2]))
    return {q: [d for _, d in sorted(v)] for q, v in out.items()}


def read_tevatron(path, qids):
    out = {}
    for line in gz_lines(path):
        m = _TEV_QID.match(line[:120].decode("utf-8", "replace"))
        if m and m.group(1) not in qids:
            continue
        row = json.loads(line)
        q = str(row["query_id"])
        if q in qids:
            out[q] = [p.get("text", "") for p in row.get("positive_passages", []) if p.get("text", "").strip()]
    return out


def read_docs(path, wanted, cache):
    if cache.exists():
        print(f"  docs from cache {cache.name}", flush=True)
        return load_pt(cache, weights_only=False)
    wb, out, t0 = {w.encode() for w in wanted}, {}, time.time()
    for i, line in enumerate(gz_lines(path)):
        did = line[:line.find(b"\t")]
        if did in wb:
            f = line.rstrip(b"\n").decode("utf-8", "replace").split("\t", 3)
            f += [""] * (4 - len(f))
            t = doc_text(f[2], f[3])
            out[f[0]] = (f[1], t[:CHAR_CUT], len(t))
            if len(out) == len(wanted):
                break
        if (i + 1) % 500_000 == 0:
            print(f"  scanned {i + 1:,} docs, found {len(out):,}/{len(wanted):,} ({time.time() - t0:.0f} s)", flush=True)
    save_pt(out, cache)
    return out


def tokenize_docs(tok, docs, ids, max_doc, info, batch=512):
    """info[did] = {win: array of window ids, n: tokens of the cut text, wtext: window text, full: text|None, bad}."""
    todo = [d for d in ids if d not in info]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        pieces = [qr.doc_piece_text(docs[d][1]) for d in chunk]
        enc = tok(pieces, add_special_tokens=False, return_offsets_mapping=True)
        if i == 0 and chunk:
            assert enc["input_ids"][:8] == qr.encode(tok, pieces[:8]), "offset tokenization differs from qr.encode"
        for d, piece, x, offs in zip(chunk, pieces, enc["input_ids"], enc["offset_mapping"]):
            url, cut, char_len = docs[d]
            was_cut = char_len > len(cut)
            k = min(len(x), max_doc)
            info[d] = {"win": array("I", x[:max_doc]), "n": len(x),
                       "wtext": piece[:offs[k - 1][1]].strip() if k else "",
                       "full": cut if (not was_cut and len(x) <= max_doc) else None,
                       "bad": (not cut.strip()) or (was_cut and len(x) < MIN_CUT_TOKENS)}
        if (i // batch) % 100 == 0:
            print(f"  tokenized {min(i + batch, len(todo)):,}/{len(todo):,}", flush=True)
    return info


def key_of(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-dev", type=int, default=1000)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--neg-pool", type=int, default=15)
    ap.add_argument("--pool", type=int, default=60000)
    ap.add_argument("--parity-queries", type=int, default=3000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)
    t0 = time.time()
    A = cfg.paths.artifacts_dir
    out_pt = A / "train" / "msmarco_doc_qwen3.pt"
    out_json = cfg.paths.results_dir / "train_data_long.json"
    cache_dir = ensure_dir(A / "train" / "msmarco_doc_cache")
    max_doc, k_neg = cfg.rerank.max_doc_tokens, args.group_size - 1

    # 1. sources
    src = sources(cfg)
    source = {k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in src.items()}

    # 2. tokenizer + pieces, asserted equal to the frozen Phase 2 / 3.1 artifacts
    tok = qr.load_tokenizer(cfg.models.qwen3.name, cfg.models.qwen3.revision)
    inst = qr.DEFAULT_INSTRUCTION
    pieces = qr.build_pieces(tok, inst)
    qr.check_tokenizer(tok, pieces)
    seam = {k: qr.encode(tok, [v])[0] for k, v in qr.seam_piece_texts().items()}
    art = load_pt(A / "tokenized" / f"{cfg.datasets[0]}_qwen3.pt")
    dfart = load_pt(A / "tokenized" / f"{cfg.datasets[0]}_qwen3_df.pt")
    assert art["pieces"] == pieces and dfart["seam_pieces"] == seam
    assert art["instruction"] == inst and art["max_doc_tokens"] == max_doc
    pieces = {**pieces, **seam}

    # 3. leakage references
    test_q, shingles = set(), set()
    for ds in cfg.datasets:
        b = load_beir(ds, split=cfg.split, data_dir=cfg.paths.data_dir, download=False)
        test_q |= {td.normalize_text(q) for q in b.queries.values()}
        for d in b.corpus.values():
            w = words(f"{d['title']} {d['text']}")
            if len(w) < SHINGLE:
                if len(w) >= 5:
                    shingles.add(hash(" ".join(w)))
                continue
            shingles.update(hash(" ".join(w[i:i + SHINGLE])) for i in range(len(w) - SHINGLE + 1))
    print(f"test queries {len(test_q)}, test doc shingles {len(shingles):,}", flush=True)

    # 4. queries, qrels, pool
    queries = read_queries(src["queries"]["path"])
    qrels = read_qrels(src["qrels"]["path"])
    eligible = sorted(q for q in qrels if q in queries)
    rng = random.Random(f"{cfg.seed}-msmarco-doc")
    rng.shuffle(eligible)
    pool = eligible[:args.pool]
    print(f"doctrain queries {len(queries):,}, with qrels {len(eligible):,}, pool {len(pool):,}", flush=True)

    # 5. Tevatron positive passages for the pool
    tev = read_tevatron(src["tevatron"]["path"], set(pool))
    print(f"pool qids in Tevatron: {len(tev):,} ({time.time() - t0:.0f} s)", flush=True)

    # 6. pass 1: positive docs, window, answer-in-window
    pos_ids = sorted({qrels[q][0] for q in pool})
    k1 = key_of([cfg.seed, args.pool, CHAR_CUT, src["docs"]["sha256"], "pass1"])
    docs = read_docs(src["docs"]["path"], set(pos_ids), cache_dir / f"pass1_{k1}.pt")
    info = tokenize_docs(tok, docs, [d for d in pos_ids if d in docs], max_doc, {})

    drops = {"leak_query": 0, "dup_query": 0, "no_passage_label": 0, "pos_doc_missing": 0, "pos_doc_bad": 0,
             "answer_outside_window": 0, "few_negatives": 0, "no_positive": 0, "leak_doc": 0, "leak_url": 0}
    n_cand = int((args.n_train + args.n_dev) * 1.25) + 50
    cands, seen_q, checked, later_in_doc = [], set(), 0, 0
    for q in pool:
        if len(cands) >= n_cand:
            break
        qn = td.normalize_text(queries[q])
        if qn in test_q:
            drops["leak_query"] += 1
            continue
        if qn in seen_q:
            drops["dup_query"] += 1
            continue
        if q not in tev:
            drops["no_passage_label"] += 1
            continue
        d = qrels[q][0]
        if d not in info:
            drops["pos_doc_missing"] += 1
            continue
        if info[d]["bad"]:
            drops["pos_doc_bad"] += 1
            continue
        checked += 1
        ww = " " + " ".join(words(info[d]["wtext"])) + " "
        if not any((pw := words(p)) and (" " + " ".join(pw) + " ") in ww for p in tev[q]):
            drops["answer_outside_window"] += 1
            cw = " " + " ".join(words(docs[d][1])) + " "
            later_in_doc += any((pw := words(p)) and (" " + " ".join(pw) + " ") in cw for p in tev[q])
            continue
        seen_q.add(qn)
        cands.append(q)
    pass_rate = (checked - drops["answer_outside_window"]) / max(checked, 1)
    print(f"candidates {len(cands):,}, answer-in-window pass rate {pass_rate:.3f} ({checked:,} checked)", flush=True)

    # 7. negatives: top-30 docs of the candidates (pass 2)
    top = read_top(src["top100"]["path"], cands, NEG_DEPTH)
    neg_ids = sorted({d for q in cands for d in top.get(q, [])} - set(docs))
    k2 = key_of([k1, CHAR_CUT, NEG_DEPTH, neg_ids[:50], len(neg_ids), "pass2"])
    docs.update(read_docs(src["docs"]["path"], set(neg_ids), cache_dir / f"pass2_{k2}.pt"))
    info = tokenize_docs(tok, docs, [d for d in neg_ids if d in docs], max_doc, info)
    print(f"negative docs {len(neg_ids):,} ({time.time() - t0:.0f} s)", flush=True)

    # 8. groups, leakage, split
    leak_cache = {}

    def leaks(d):
        if d not in leak_cache:
            url, cut, _ = docs[d]
            w = words(cut)
            hit = any(hash(" ".join(w[i:i + SHINGLE])) in shingles for i in range(len(w) - SHINGLE + 1))
            leak_cache[d] = ("url" if any(x in url.lower() for x in LEAK_DOMAINS) else None) or ("doc" if hit else None)
        return leak_cache[d]

    split = {"dev": [], "train": []}
    query_text, groups, hard, used = {}, {}, [], set()
    for q in cands:
        if len(split["dev"]) >= args.n_dev and len(split["train"]) >= args.n_train:
            break
        pos = qrels[q][0]
        negs = [d for d in top.get(q, []) if d in info and not info[d]["bad"]]
        row = {"query_id": q, "query": queries[q],
               "positive_passages": [{"docid": pos, "title": "", "text": info[pos]["wtext"]}],
               "negative_passages": [{"docid": d, "title": "", "text": info[d]["wtext"]} for d in negs]}
        g, tx, h = td.select_group(row, args.group_size, args.neg_pool, cfg.seed)
        if g is None:
            drops[tx] += 1
            continue
        lk = [leaks(d) for d in g]
        if "url" in lk:
            drops["leak_url"] += 1
            continue
        if "doc" in lk:
            drops["leak_doc"] += 1
            continue
        which = "dev" if len(split["dev"]) < args.n_dev else "train"
        split[which].append(q)
        query_text[q] = queries[q]
        groups[q] = g
        used.update(g)
        hard.append(h)
    if len(split["train"]) < args.n_train or len(split["dev"]) < args.n_dev:
        sys.exit(f"not enough groups: train {len(split['train'])}, dev {len(split['dev'])}; raise --pool")

    # 9. tokenize queries; docs from the window ids
    qids = split["dev"] + split["train"]
    q_qf = dict(zip(qids, qr.encode(tok, [qr.query_text_qf(query_text[i]) for i in qids])))
    q_df = dict(zip(qids, qr.encode(tok, [qr.query_text_df(query_text[i]) for i in qids])))
    pids = sorted(used)
    doc, doc_seam, seam_kind, full_len = {}, {}, {}, {}
    for p in pids:
        doc[p] = list(info[p]["win"])
        full_len[p] = info[p]["n"]
        s, kind = qr.seam_doc_ids(tok, doc[p], text=info[p]["full"])
        assert len(s) <= max_doc + 1, (p, len(s))
        doc_seam[p], seam_kind[p] = s, kind

    special = set(tok.all_special_ids)
    bad = [p for p in pids if special & set(doc_seam[p])] + [i for i in qids if special & set(q_qf[i])]
    if bad:
        sys.exit(f"special token ids inside text for {len(bad)} items, e.g. {bad[:5]}")

    # 10. parity on untruncated pairs (few for documents)
    par_q = split["dev"] + split["train"][: args.parity_queries]
    qs, ds_, kinds_, ours_qf, ours_df = [], [], [], [], []
    for qid in par_q:
        for p in groups[qid]:
            if info[p]["full"] is not None:
                qs.append(query_text[qid])
                ds_.append(info[p]["full"])
                kinds_.append(seam_kind[p])
                ours_qf.append(qr.query_first_ids(pieces, q_qf[qid], doc[p]))
                ours_df.append(qr.doc_first_seam_ids(pieces, doc_seam[p], q_df[qid]))
    ref_qf, ref_df = [], []
    for i in range(0, len(qs), 512):
        ref_qf += qr.official_query_first_ids(tok, inst, qs[i:i + 512], ds_[i:i + 512])
        ref_df += qr.natural_doc_first_ids(tok, inst, qs[i:i + 512], ds_[i:i + 512])
    by_kind = {}
    for k, a, b in zip(kinds_, ours_df, ref_df):
        e = by_kind.setdefault(k, {"pairs": 0, "match": 0})
        e["pairs"] += 1
        e["match"] += int(a == b)
    parity = {"pairs": len(qs), "query_first_match": sum(a == b for a, b in zip(ours_qf, ref_qf)),
              "doc_first_seam_match": sum(a == b for a, b in zip(ours_df, ref_df)), "doc_first_by_kind": by_kind}

    # 11. stats (+ Phase 4 reference)
    seq_qf = [len(qr.query_first_ids(pieces, q_qf[q], doc[p])) for q in qids for p in groups[q]]
    seq_df = [len(qr.doc_first_seam_ids(pieces, doc_seam[p], q_df[q])) for q in qids for p in groups[q]]
    dev_p = {p for q in split["dev"] for p in groups[q]}
    train_p = {p for q in split["train"] for p in groups[q]}
    kinds = {}
    for k in seam_kind.values():
        kinds[k] = kinds.get(k, 0) + 1
    p4 = load_json(cfg.paths.results_dir / "train_data.json")["stats"]
    stats = {
        "doctrain_queries": len(queries), "eligible": len(eligible), "pool": len(pool), "pool_in_tevatron": len(tev),
        "answer_in_window": {"checked": checked, "pass_rate": round(pass_rate, 4),
                             "outside_but_in_cut_text": later_in_doc,
                             "not_found_in_cut_text": drops["answer_outside_window"] - later_in_doc},
        "drops": drops, "num_train": len(split["train"]), "num_dev": len(split["dev"]), "num_docs": len(pids),
        "dev_docs_also_in_train": len(dev_p & train_p),
        "doc_tokens_window": len_stats(len(doc[p]) for p in pids),
        "doc_tokens_cut_text": len_stats(full_len.values()),
        "doc_chars_full": len_stats(docs[p][2] for p in pids),
        "truncated_frac": round(sum(len(doc[p]) == max_doc and (info[p]["full"] is None) for p in pids) / len(pids), 4),
        "query_tokens_df": len_stats(len(q_df[q]) for q in qids),
        "seq_len_qf": len_stats(seq_qf), "seq_len_df": len_stats(seq_df), "seam_kind": kinds,
        "hardness_mean": {k: round(sum(h[k] for h in hard) / len(hard), 4) for k in hard[0]},
        "parity": parity,
        "phase4_reference": {k: p4[k] for k in ("passage_tokens_full", "truncated_frac", "seq_len_qf", "seq_len_df",
                                                 "hardness_mean")},
    }

    # 12. content + guard + save (same keys as 06a)
    content = {
        "dataset": "msmarco-document", "source": source,
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
    save_pt({**content, "content_sha256": sha, "run_info": run_info()}, out_pt)
    save_json({"run_info": run_info(), "rules": RULES, "source": source, "content_sha256": sha,
               "settings": vars(args), "stats": stats}, out_json)

    print(f"train {stats['num_train']}  dev {stats['num_dev']}  docs {stats['num_docs']}")
    print(f"drops {drops}")
    print(f"answer-in-window {stats['answer_in_window']}")
    print(f"window tokens {stats['doc_tokens_window']}  truncated {stats['truncated_frac']}")
    print(f"seq qf {stats['seq_len_qf']}\nseq df {stats['seq_len_df']}")
    print(f"phase 4 seq df {p4['seq_len_df']}")
    print(f"seam kinds {kinds}")
    print(f"hardness {stats['hardness_mean']}  (phase 4 {p4['hardness_mean']})")
    print(f"parity {parity}")
    print(f"sha {sha}  ({time.time() - t0:.0f} s)  -> {out_pt}")


if __name__ == "__main__":
    main()