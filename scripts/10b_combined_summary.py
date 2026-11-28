# scripts/10b_combined_summary.py
"""Phase 7.4: exact packed sizes of combined caches, one table of every Phase 5 / 6 / 7 point, frontier.

Sizes: kv.pack.predicted_file_bytes(fmt, kept, shared_len, next_pos=shared_len + T) per doc from the 7.3
score files' retention_per_doc; corpus = sum + one shared-prefix file (5.5). In-run check arms are sized
the same way and must equal the Phase 5 / 6 sizes. Every single-method delta and verdict is recomputed
from its score file against p5_b8 and must equal the stored verdict. A stratified sample of real docs is
built, compressed, packed and unpacked bit-exact, file size == prediction. Frontier: the rule
pre-registered in 10_combined_sweep.py RULES["frontier_7_4"].
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

from marginkv.config import load_config
from marginkv.data.beir_loader import load_beir
from marginkv.eval import paired
from marginkv.eval.rerank import evaluate_rerank
from marginkv.kv import cache_ops as co
from marginkv.kv import combined as CB
from marginkv.kv import pack
from marginkv.kv import sweep
from marginkv.models import qwen3_reranker as qr
from marginkv.train.data import sha256_file
from marginkv.utils.device import get_device, get_dtype
from marginkv.utils.io import load_json, load_pt, run_info, save_csv, save_json

PHASE = "7.4"
MIB, GIB = 2 ** 20, 2 ** 30
P7 = "qwen3_lora_dfB_cached_p7_{arm}_{ds}.json"
P6 = "qwen3_lora_dfB_cached_p6_{arm}_{ds}.json"
P5Q = "qwen3_lora_dfB_cached_q_{fmt}_{ds}.json"
P5B = "qwen3_lora_dfB_cached_p5_b8_{ds}.json"
COMBINED = ("uniform_r75_fp8_tensor", "uniform_r75_int4_kivi")
CHECK_SIZES = {"uniform_r75": "p6:uniform_r75", "fp8_tensor": "p5:fp8_tensor", "int4_kivi": "p5:int4_kivi"}
REL_TOL = 1e-9


def verify_model(model_dir):
    shas = load_json(model_dir / "marginkv_merge.json")["files_sha256"]
    for fname, want in shas.items():
        if sha256_file(model_dir / fname) != want:
            raise SystemExit(f"sha mismatch: {fname}")
    return shas


def arm_bytes(rpd, fmt, s):
    return np.array([pack.predicted_file_bytes(fmt, int(k), s, next_pos=s + int(t)) for t, k in rpd.values()],
                    dtype=np.float64)


def same_kv(a, b):
    if a.next_pos != b.next_pos or a.shared_len != b.shared_len or a.num_layers != b.num_layers:
        return False
    return all(torch.equal(a.keys[l], b.keys[l]) and torch.equal(a.values[l], b.values[l]) for l in range(a.num_layers))


def pack_sample(model, device, art, dfart, recs, n):
    pieces, _, _ = sweep.sweep_plan(art, dfart)
    s = co.shared_prefix_len(pieces)
    rpd0 = recs[COMBINED[0]]["retention_per_doc"]
    order = sorted(rpd0, key=lambda d: (rpd0[d][0], d))
    pick = sorted({order[round(i * (len(order) - 1) / max(n - 1, 1))] for i in range(n)})
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for d in pick:
            prefix, _ = co.split_doc_first(pieces, dfart["doc_seam"][d], [], variant="B")
            kv = co.build_doc_kvs(model, [prefix], device, s)[0].to("cpu")
            for arm in COMBINED:
                rec = recs[arm]
                out = CB.make_combined(rec["method"], rec["retention"], rec["format"])(kv, d)
                path = Path(tmp) / f"{arm}_{d}.safetensors"
                info = pack.pack_doc(out, rec["format"], path)
                pred = pack.predicted_file_bytes(rec["format"], out.n_cached - s, s, next_pos=out.next_pos)
                back = pack.unpack_doc(path, [k[:, :s] for k in out.keys], [v[:, :s] for v in out.values])
                rows.append({"doc": d, "arm": arm, "doc_tokens": rpd0[d][0], "kept": out.n_cached - s,
                             "kept_matches_sweep": out.n_cached - s == rec["retention_per_doc"][d][1],
                             "file_bytes": info["file_bytes"], "predicted": pred, "on_disk": path.stat().st_size,
                             "size_match": info["file_bytes"] == pred == path.stat().st_size,
                             "bit_exact": same_kv(back, out)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=4, help="real docs per dataset for the pack round trip")
    ap.add_argument("--skip-pack", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    torch.set_grad_enabled(False)
    A, R = cfg.paths.artifacts_dir, cfg.paths.results_dir
    S, M = R / "scores", R / "metrics"
    depth = cfg.rerank.depth
    storage_all = load_json(M / "storage.json")["datasets"]
    quant_all = load_json(M / "quant.json")["datasets"]
    ps_all = load_json(M / "prune_summary.json")["datasets"]
    comb_report = load_json(M / "combined.json")
    comb_all = comb_report["datasets"]
    rule = comb_report["rules"]["frontier_7_4"]

    model, device = None, None
    if not args.skip_pack:
        device, dtype = get_device(cfg.runtime.device), get_dtype(cfg.runtime.dtype)
        model_dir = A / "lora" / "df" / "merged"
        verify_model(model_dir)
        model = qr.load_model(str(model_dir), None, device, dtype)

    out = {"run_info": run_info(), "phase": PHASE, "rule": rule, "rules_sha256": comb_report["rules_sha256"],
           "datasets": {}}
    pqs, table = {}, []

    for ds in cfg.datasets:
        cj = load_json(R / "candidates" / f"{ds}.json")
        qids, cands = list(cj["query_order"]), cj["candidates"]
        beir = load_beir(ds, cfg.split, data_dir=str(cfg.paths.data_dir), download=False)
        qrels = {q: beir.qrels[q] for q in qids}
        ev = lambda p: evaluate_rerank(cands, load_json(p)["scores"], qrels, depth)["per_query"]
        st, qf, ps, cb = storage_all[ds], quant_all[ds]["formats"], ps_all[ds]["arms"], comb_all[ds]
        margin, s, shared_file = cb["margin"], st["shared_len"], st["shared_file_bytes"]
        bf16_mib = st["formats"]["bf16"]["file_mib_per_doc_mean"]
        bf16_corpus = st["formats"]["bf16"]["corpus_total_bytes"]
        base_pq = ev(S / P5B.format(ds=ds))
        P, pq = {}, {}

        def add(pid, kind, method, retention, fmt, diag, this_pq, mib, corpus_bytes, stored_verdict):
            if this_pq is base_pq:
                c, verdict = None, "baseline"
            else:
                c = paired.compare(base_pq, this_pq, qids, cfg.seed)
                verdict = paired.verdict(c["delta_ndcg@10"], c["ndcg@10_ci95"], margin)
                assert verdict == stored_verdict, f"[{ds}] {pid}: verdict {verdict} != stored {stored_verdict}"
            P[pid] = {"kind": kind, "method": method, "retention": retention, "format": fmt, "diagnostic": diag,
                      "mib_per_doc": mib, "ratio_vs_bf16": bf16_mib / mib, "corpus_gib": corpus_bytes / GIB,
                      "ratio_vs_bf16_corpus": bf16_corpus / corpus_bytes,
                      "d_ndcg10": c["delta_ndcg@10"] if c else 0.0, "ndcg10_ci": c["ndcg@10_ci95"] if c else None,
                      "d_mrr10": c["delta_mrr@10"] if c else 0.0, "verdict": verdict}
            pq[pid] = this_pq

        for fmt, e in st["formats"].items():
            this = base_pq if fmt == "bf16" else ev(S / P5Q.format(fmt=fmt, ds=ds))
            add(f"q:{fmt}", "quant", None, 1.0, fmt, False, this, e["file_mib_per_doc_mean"], e["corpus_total_bytes"],
                None if fmt == "bf16" else qf[fmt]["verdict"])
        for arm, e in ps.items():
            add(f"p:{arm}", "prune", e["method"], e["retention"], "bf16", e["diagnostic"], ev(S / P6.format(arm=arm, ds=ds)),
                e["mib_per_doc"], e["corpus_gib"] * GIB, e["verdict"])

        size_checks = {}
        recs = {a: load_json(S / P7.format(arm=a, ds=ds)) for a in (*COMBINED, *CHECK_SIZES)}
        for a, rec in recs.items():
            b = arm_bytes(rec["retention_per_doc"], rec["format"], s)
            mib, corpus = float(b.mean() / MIB), float(b.sum() + shared_file)
            if a in CHECK_SIZES:
                src, ref_id = CHECK_SIZES[a].split(":")
                ref = P[f"{'p' if src == 'p6' else 'q'}:{ref_id}"]["mib_per_doc"]
                rel = abs(mib - ref) / ref
                size_checks[a] = {"mib": mib, "ref_mib": ref, "rel_diff": rel, "match": bool(rel <= REL_TOL)}
                assert size_checks[a]["match"], f"[{ds}] {a}: size {mib:.6f} vs ref {ref:.6f}"
            else:
                add(f"c:{a}", "combined", rec["method"], rec["retention"], rec["format"], False,
                    ev(S / P7.format(arm=a, ds=ds)), mib, corpus, cb["arms"][a]["verdict"])

        pts = {k: v for k, v in P.items() if not v["diagnostic"]}
        for k, v in pts.items():
            v["point_frontier"] = not any(
                j != k and w["mib_per_doc"] <= v["mib_per_doc"] and w["d_ndcg10"] >= v["d_ndcg10"]
                and (w["mib_per_doc"] < v["mib_per_doc"] or w["d_ndcg10"] > v["d_ndcg10"]) for j, w in pts.items())
        for v in P.values():
            v.setdefault("point_frontier", None)

        sample = None
        if model is not None:
            art = load_pt(A / "tokenized" / f"{ds}_qwen3.pt")
            dfart = load_pt(A / "tokenized" / f"{ds}_qwen3_df.pt")
            rows = pack_sample(model, device, art, dfart, recs, args.sample)
            sample = {"rows": rows, "n": len(rows),
                      "all_ok": all(r["size_match"] and r["bit_exact"] and r["kept_matches_sweep"] for r in rows)}
            assert sample["all_ok"], f"[{ds}] pack sample failed: {[r for r in rows if not (r['size_match'] and r['bit_exact'])]}"

        out["datasets"][ds] = {"num_queries": len(qids), "margin": margin, "shared_len": s,
                               "shared_file_bytes": shared_file, "bf16_mib_per_doc": bf16_mib,
                               "size_checks": size_checks, "pack_sample": sample, "points": P}
        pqs[ds] = pq
        for k, v in P.items():
            ci = v["ndcg10_ci"] or [None, None]
            table.append({"dataset": ds, "id": k, "kind": v["kind"], "method": v["method"], "retention": v["retention"],
                          "format": v["format"], "diagnostic": v["diagnostic"], "d_ndcg10": v["d_ndcg10"],
                          "ci_lo": ci[0], "ci_hi": ci[1], "d_mrr10": v["d_mrr10"], "verdict": v["verdict"],
                          "mib_per_doc": v["mib_per_doc"], "ratio_vs_bf16": v["ratio_vs_bf16"],
                          "corpus_gib": v["corpus_gib"], "point_frontier": v["point_frontier"]})

        print(f"[{ds}] size checks: " + " ".join(f"{a} rel {c['rel_diff']:.1e}" for a, c in size_checks.items())
              + f" | pack sample: {sample['n'] if sample else 'skipped'} files, ok {sample['all_ok'] if sample else '-'}")
        for k in sorted(pts, key=lambda k: pts[k]["mib_per_doc"]):
            v = pts[k]
            ci = v["ndcg10_ci"]
            cs = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else " " * 18
            print(f"[{ds}] {k:28s} {v['mib_per_doc']:6.2f} MiB {v['ratio_vs_bf16']:5.2f}x  d {v['d_ndcg10']:+.4f} {cs} "
                  f"{v['verdict']:18s} {'*' if v['point_frontier'] else ''}")

    dss = list(out["datasets"])
    D = {ds: out["datasets"][ds]["points"] for ds in dss}
    singles = [k for k, v in D[dss[0]].items() if v["kind"] in ("quant", "prune") and not v["diagnostic"]]
    frontier = {}
    for a in COMBINED:
        c = f"c:{a}"
        smaller = [x for x in singles if all(D[ds][x]["mib_per_doc"] <= D[ds][c]["mib_per_doc"] for ds in dss)]
        comp, dominated_by, beats = {}, [], []
        for x in smaller:
            xc = {ds: paired.compare(pqs[ds][c], pqs[ds][x], qids_ds, cfg.seed)
                  for ds, qids_ds in ((ds, list(pqs[ds][c])) for ds in dss)}
            comp[x] = {ds: [xc[ds]["delta_ndcg@10"], xc[ds]["ndcg@10_ci95"]] for ds in dss}  # x - c
            if all(xc[ds]["ndcg@10_ci95"][0] > 0 for ds in dss):
                dominated_by.append(x)
            if all(xc[ds]["ndcg@10_ci95"][1] < 0 for ds in dss):
                beats.append(x)
        label = "dominated" if dominated_by else ("on_frontier" if len(beats) == len(smaller) else "not_separated")
        frontier[a] = {"label": label, "n_smaller": len(smaller), "vacuous": len(smaller) == 0,
                       "smaller": smaller, "dominated_by": dominated_by, "beats": beats, "x_minus_c": comp,
                       "point_frontier": {ds: D[ds][c]["point_frontier"] for ds in dss}}
        print(f"[frontier] {a:23s} {label:14s} n_smaller {len(smaller)} vacuous {len(smaller) == 0} "
              f"dominated_by {dominated_by} beats {len(beats)}/{len(smaller)} point_frontier {frontier[a]['point_frontier']}")
        for x, v in comp.items():
            print(f"           {x:28s} x - c " + "  ".join(f"{ds} {d:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]" for ds, (d, ci) in v.items()))
    out["frontier"] = frontier

    save_json(out, M / "combined_summary.json")
    save_csv(table, M / "combined_table.csv")
    print(f"wrote {M / 'combined_summary.json'} and {M / 'combined_table.csv'} ({len(table)} rows)")


if __name__ == "__main__":
    main()