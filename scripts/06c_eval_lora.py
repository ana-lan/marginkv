# scripts/06c_eval_lora.py
"""Phase 4.4: BEIR order gate for the LoRA models (full prompt, no KV cache).

Arms:
  qf  = query-first LoRA control (artifacts/lora/qf/merged), query-first prompt
  dfB = doc-first LoRA (artifacts/lora/df/merged), variant B prompt
Inputs are asserted byte-identical to the Phase 2 / 3.2 runs (qwen3_{ds}.json, qwen3_dfB_{ds}.json).
Same batch size, length-sorted fixed batches, MPS bf16. Resumable; --smoke N under artifacts/smoke/.
Requires results/metrics/lora_merge_check.json to pass for the arm.

Outputs:
  results/scores/qwen3_lora_{qf,dfB}_{ds}.json
  results/metrics/lora.json
Order gate (pre-registered): lora_dfB nDCG@10 >= lora_qf - 0.01 on every dataset.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import gc
import hashlib
import json
import time

import numpy as np
import torch

from marginkv.config import load_config
from marginkv.data.beir_loader import load_beir
from marginkv.eval.rerank import evaluate_rerank, identity_scores
from marginkv.models import qwen3_reranker as qr
from marginkv.train import data as td
from marginkv.utils.device import get_device, get_dtype
from marginkv.utils.io import ensure_dir, load_json, load_pt, run_info, save_json
from marginkv.utils.seed import set_seed
from marginkv.utils.timing import Timer, summarize_ms

CHUNK_BATCHES = 16
KEYS = ("log_odds", "p_yes", "logit_yes", "logit_no")
GATE = 0.01
BOOT = 1000
EPS = 1e-9
ARMS = {  # arm -> (lora dir, reference file whose inputs must match, prompt order)
    "qf": ("qf", "qwen3_{ds}.json", "query_first"),
    "dfB": ("df", "qwen3_dfB_{ds}.json", "doc_first"),
}


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_inputs(art, dfart, arm, qids):
    top = art["rerank_candidates"]
    pairs = [(q, d) for q in qids for d in top[q]]
    if arm == "qf":
        seqs = [qr.query_first_ids(art["pieces"], art["query_qf"][q], art["doc"][d]) for q, d in pairs]
    else:
        pb = {**art["pieces"], **dfart["seam_pieces"]}
        seqs = [qr.doc_first_seam_ids(pb, dfart["doc_seam"][d], art["query_df"][q]) for q, d in pairs]
    return pairs, seqs


def score_run(model, seqs, batch_size, device, partial_path, meta, tag):
    """Same rule as 04: length-sorted fixed batches, state saved every CHUNK_BATCHES."""
    order = sorted(range(len(seqs)), key=lambda i: (len(seqs[i]), i))
    batches = [order[s:s + batch_size] for s in range(0, len(order), batch_size)]
    state = {"meta": meta, "batches": {}, "batch_ms": {}}
    if partial_path.exists():
        old = load_json(partial_path)
        if old.get("meta") != meta:
            raise SystemExit(f"{tag} partial {partial_path} has different inputs/settings. Delete it or use --force.")
        state = old
        print(f"{tag} resuming: {len(state['batches'])}/{len(batches)} batches done", flush=True)
    todo = [b for b in range(len(batches)) if str(b) not in state["batches"]]
    left = sum(len(batches[b]) for b in todo)
    t0, done = time.time(), 0
    for c in range(0, len(todo), CHUNK_BATCHES):
        for b in todo[c:c + CHUNK_BATCHES]:
            with Timer(device) as t:
                res = qr.score_batch(model, [seqs[i] for i in batches[b]], device)
            state["batches"][str(b)] = {k: res[k] for k in KEYS}
            state["batch_ms"][str(b)] = t.ms
            done += len(batches[b])
        save_json(state, partial_path)
        rate = done / (time.time() - t0)
        print(f"{tag} {len(state['batches'])}/{len(batches)} batches  {rate:.2f} pairs/s  "
              f"ETA {(left - done) / rate / 60:.1f} min", flush=True)
    out = {k: [None] * len(seqs) for k in KEYS}
    for b, idx in enumerate(batches):
        r = state["batches"][str(b)]
        for k in KEYS:
            for j, i in enumerate(idx):
                out[k][i] = r[k][j]
    return out, [state["batch_ms"][str(b)] for b in range(len(batches))]


def _paired(x, y, qids, seed) -> dict:
    dn = np.array([x["per_query"][q]["ndcg@10"] - y["per_query"][q]["ndcg@10"] for q in qids])
    dm = np.array([x["per_query"][q]["mrr@10"] - y["per_query"][q]["mrr@10"] for q in qids])
    rng = np.random.default_rng(seed)
    boot = dn[rng.integers(0, len(qids), size=(BOOT, len(qids)))].mean(1)
    return {"delta_ndcg@10": float(dn.mean()), "delta_mrr@10": float(dm.mean()),
            "ndcg@10_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "wlt_ndcg@10": [int((dn > EPS).sum()), int((dn < -EPS).sum()), int((np.abs(dn) <= EPS).sum())]}


def _agreement(s, ref, qids) -> dict:
    rhos, top1 = [], 0
    for q in qids:
        docs = list(ref[q])
        a = np.array([ref[q][d] for d in docs])
        b = np.array([s[q][d] for d in docs])
        top1 += docs[int(a.argmax())] == docs[int(b.argmax())]
        rhos.append(float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1]))
    return {"spearman_mean": float(np.mean(rhos)), "top1_agree_frac": top1 / len(qids)}


def compute(ds, cand, qids, systems, depth, seed, smoke) -> dict:
    cs = {q: cand["candidates"][q] for q in qids}
    qrels = {q: ds.qrels[q] for q in qids}
    ev = {k: evaluate_rerank(cs, {q: s[q] for q in qids}, qrels, depth) for k, s in systems.items()}
    r100 = ev["bm25"]["mean"]["recall@100"]
    for k, e in ev.items():
        if abs(e["mean"]["recall@100"] - r100) > 1e-12:
            raise SystemExit(f"{k}: Recall@100 changed after reranking")
    out = {k: {"mean": e["mean"], "num_queries": e["num_queries"], "diagnostics": e["diagnostics"],
               "per_query": e["per_query"]} for k, e in ev.items()}
    if "lora_dfB" in ev and "lora_qf" in ev:
        g = _paired(ev["lora_dfB"], ev["lora_qf"], qids, seed)
        out["gate"] = {**g, "agreement": _agreement(systems["lora_dfB"], systems["lora_qf"], qids),
                       "threshold": GATE, "pass": None if smoke else bool(g["delta_ndcg@10"] >= -GATE)}
    for a, b, key in (("lora_dfB", "qwen3_qf", "dfB_vs_orig_qf"),
                      ("lora_qf", "qwen3_qf", "control_vs_orig_qf"),
                      ("lora_dfB", "qwen3_dfB_zs", "dfB_lora_vs_zeroshot")):
        if a in ev and b in ev:
            out[key] = _paired(ev[a], ev[b], qids, seed)
    if "control_vs_orig_qf" in out:
        out["control_vs_orig_qf"]["control_drop_flag"] = out["control_vs_orig_qf"]["delta_ndcg@10"] < -GATE
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["dfB", "qf"], choices=list(ARMS))
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)
    depth, bs = cfg.rerank.depth, cfg.rerank.batch_size
    smoke = args.smoke > 0
    R, A = cfg.paths.results_dir, cfg.paths.artifacts_dir
    score_dir = ensure_dir(A / "smoke" / "scores" if smoke else R / "scores")
    partial_dir = ensure_dir(A / ("smoke_partial" if smoke else "scores_partial"))
    metrics_path = ensure_dir(A / "smoke" / "metrics" if smoke else R / "metrics") / "lora.json"
    device, dtype = get_device(cfg.runtime.device), get_dtype(cfg.runtime.dtype)
    info = run_info()

    mc_path = R / "metrics" / "lora_merge_check.json"
    if not mc_path.exists():
        raise SystemExit("run scripts/06b_merge_check.py first")
    mc = load_json(mc_path)["arms"]

    data = {}
    for name in cfg.datasets:
        art = load_pt(A / "tokenized" / f"{name}_qwen3.pt")
        dfart = load_pt(A / "tokenized" / f"{name}_qwen3_df.pt")
        if dfart["base_content_sha256"] != art["content_sha256"]:
            raise SystemExit(f"[{name}] seam artifact does not match the Phase 2 artifact")
        qids = art["query_order"][:args.smoke] if smoke else art["query_order"]
        data[name] = (art, dfart, qids)

    lora_scores = {name: {} for name in cfg.datasets}
    for arm in args.arms:
        ldir, ref_tpl, porder = ARMS[arm]
        if not mc[ldir]["pass_merge_only"]:
            raise SystemExit(f"[{arm}] merge check failed; evaluate with the unmerged adapter instead (ask first)")
        mdir = A / "lora" / ldir / "merged"
        merge_info = load_json(mdir / "marginkv_merge.json")
        files = {p: td.sha256_file(mdir / p) for p in merge_info["files_sha256"]}
        if files != merge_info["files_sha256"]:
            raise SystemExit(f"[{arm}] merged weights do not match marginkv_merge.json")
        model = None
        for name in cfg.datasets:
            art, dfart, qids = data[name]
            tag = f"[{name} lora_{arm}]"
            pairs, seqs = build_inputs(art, dfart, arm, qids)
            in_sha = _sha(seqs)
            if not smoke:
                ref = load_json(R / "scores" / ref_tpl.format(ds=name))
                if ref["inputs_sha256"] != in_sha:
                    raise SystemExit(f"{tag} inputs differ from {ref_tpl.format(ds=name)}")
            path = score_dir / f"qwen3_lora_{arm}_{name}.json"
            if path.exists() and not args.force:
                old = load_json(path)
                if old["inputs_sha256"] == in_sha and old["batch_size"] == bs:
                    print(f"{tag} reusing {path}")
                    lora_scores[name][arm] = old["scores"]
                    continue
                raise SystemExit(f"{tag} {path} was scored on different inputs. Rerun with --force.")
            if model is None:
                print(f"[lora_{arm}] loading {mdir}", flush=True)
                model = qr.load_model(str(mdir), None, device, dtype)
            meta = {"dataset": name, "arm": arm, "inputs_sha256": in_sha, "num_pairs": len(seqs),
                    "batch_size": bs, "chunk_batches": CHUNK_BATCHES, "device": str(device),
                    "dtype": cfg.runtime.dtype, "model_files_sha256": files}
            ppath = partial_dir / f"qwen3_lora_{arm}_{name}.json"
            if args.force and ppath.exists():
                ppath.unlink()
            out, ms = score_run(model, seqs, bs, device, ppath, meta, tag)
            if not np.all(np.isfinite(out["log_odds"])):
                raise SystemExit(f"{tag} non-finite scores. Paste the log.")
            nest = {k: {} for k in KEYS}
            for (q, d), *vals in zip(pairs, *(out[k] for k in KEYS)):
                for k, x in zip(KEYS, vals):
                    nest[k].setdefault(q, {})[d] = round(float(x), 6)
            total_s = sum(ms) / 1000
            save_json({
                "run_info": info, "model": f"qwen3_lora_{arm}", "arm": arm, "prompt_order": porder,
                "base": {"name": cfg.models.qwen3.name, "revision": cfg.models.qwen3.revision},
                "model_dir": str(mdir.relative_to(cfg.paths.artifacts_dir.parent)),
                "model_files_sha256": files, "merge": merge_info.get("adapter"),
                "dataset": name, "num_pairs": len(seqs), "batch_size": bs, "chunk_batches": CHUNK_BATCHES,
                "device": str(device), "dtype": cfg.runtime.dtype, "inputs_sha256": in_sha, "depth": depth,
                "smoke_queries": args.smoke or None, "score_key": "log_odds", "aux_key": "p_yes",
                "timing": {"total_s": round(total_s, 2), "pairs_per_s": round(len(seqs) / total_s, 3),
                           "batch_ms": summarize_ms(ms)},
                "query_order": qids, "scores": nest["log_odds"], "aux": nest["p_yes"],
                "logit_yes": nest["logit_yes"], "logit_no": nest["logit_no"],
            }, path)
            ppath.unlink(missing_ok=True)
            print(f"{tag} saved {path}  ({len(seqs) / total_s:.2f} pairs/s)", flush=True)
            lora_scores[name][arm] = nest["log_odds"]
        if model is not None:
            del model
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()

    per_ds = {}
    for name in cfg.datasets:
        art, dfart, qids = data[name]
        ds = load_beir(name, split=cfg.split, data_dir=cfg.paths.data_dir)
        cand = load_json(R / "candidates" / f"{name}.json")
        cs = {q: cand["candidates"][q] for q in qids}
        systems = {"bm25": identity_scores(cs, depth),
                   "qwen3_qf": load_json(R / "scores" / f"qwen3_{name}.json")["scores"],
                   "qwen3_dfB_zs": load_json(R / "scores" / f"qwen3_dfB_{name}.json")["scores"]}
        for arm, s in lora_scores[name].items():
            systems[f"lora_{arm}"] = s
        per_ds[name] = compute(ds, cand, qids, systems, depth, cfg.seed, smoke)

    gates = [m["gate"]["pass"] for m in per_ds.values() if "gate" in m]
    overall = None if smoke or len(gates) != len(cfg.datasets) else all(gates)
    save_json({"run_info": info, "depth": depth, "smoke_queries": args.smoke or None, "gate_threshold": GATE,
               "bootstrap_reps": BOOT, "merge_check": mc, "order_gate_pass_all": overall,
               "datasets": per_ds}, metrics_path)

    print(f"\nmetrics -> {metrics_path}" + ("   (SMOKE: pipeline check only)" if smoke else ""))
    for name, m in per_ds.items():
        print(f"\n{name}")
        for k in ("bm25", "qwen3_qf", "qwen3_dfB_zs", "lora_qf", "lora_dfB"):
            if k in m:
                print(f"  {k:13s} nDCG@10 {m[k]['mean']['ndcg@10']:.4f}  MRR@10 {m[k]['mean']['mrr@10']:.4f}")
        for key, label in (("gate", "dfB - control"), ("dfB_vs_orig_qf", "dfB - orig qf"),
                           ("control_vs_orig_qf", "control - orig"), ("dfB_lora_vs_zeroshot", "dfB lora - zs")):
            if key in m:
                p = m[key]
                ci = f"[{p['ndcg@10_ci95'][0]:+.4f},{p['ndcg@10_ci95'][1]:+.4f}]"
                extra = ""
                if key == "gate":
                    g = {None: "-", True: "PASS", False: "FAIL"}[p["pass"]]
                    extra = f"  rho {p['agreement']['spearman_mean']:.2f}  top1 {p['agreement']['top1_agree_frac']:.2f}  {g}"
                if key == "control_vs_orig_qf" and p["control_drop_flag"]:
                    extra = "  FLAG: control > 0.01 below original"
                print(f"  {label:15s} dNDCG {p['delta_ndcg@10']:+.4f} {ci}  W/L/T {'/'.join(map(str, p['wlt_ndcg@10']))}{extra}")
    print(f"\norder gate, all datasets: {overall}")


if __name__ == "__main__":
    main()