# scripts/14_export_app_data.py
"""Phase 9b: export small, view-shaped JSON for the results explorer app.

Reads committed results only (results/metrics/{lora, quant, storage, trace_stats, vllm}.json)
and writes app/data/{overview, quant, storage, trace3}.json. Every number on the site comes
from these files. Each output carries provenance (source paths, sha256, source git commits).
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from marginkv.config import REPO_ROOT, load_config
from marginkv.kv.pack import predicted_file_bytes
from marginkv.kv.quant import FORMATS, GROUP
from marginkv.utils.io import ensure_dir, load_json, run_info

SOURCES = ("lora", "quant", "storage", "trace_stats", "vllm")
DS_LABEL = {"scifact": "SciFact", "nfcorpus": "NFCorpus"}

SYSTEMS = [  # (key in lora.json, label, note)
    ("bm25", "BM25", "first stage, top 100"),
    ("qwen3_qf", "Original query-first", "Qwen3-Reranker-0.6B as released"),
    ("lora_qf", "Query-first LoRA (control)", "same MS MARCO training, original order"),
    ("qwen3_dfB_zs", "Doc-first zero-shot", "document first, no training"),
    ("lora_dfB", "Doc-first LoRA", "document first, KV reusable across queries"),
]

FORMAT_INFO = {  # id -> (label, family, note)
    "bf16": ("BF16", "bf16", "uncompressed baseline, shared prefix stored once"),
    "fp8_tensor": ("FP8 per-tensor", "fp8", "e4m3, one scale per tensor"),
    "fp8_head": ("FP8 per-head", "fp8", "e4m3, one scale per KV head"),
    "int8_tok": ("INT8 per-token", "int8", "symmetric, one scale per head and token"),
    "int4_kivi": ("INT4 KIVI", "int4", "keys per channel, values per token, groups of 32"),
    "int4_tok": ("INT4 per-token", "int4", "keys and values per token, groups of 32 (diagnostic)"),
}
HEADLINE_FORMAT = "int4_kivi"
VERDICT_LABEL = {
    "baseline": "Baseline",
    "no_measurable_loss": "No measurable loss",
    "within_1_point": "Within 1 point",
    "loss": "Loss",
}

SETTINGS = [  # vLLM settings in display order
    ("qf", "Query-first (original model)"),
    ("df", "Doc-first, BF16 KV"),
    ("df_fp8", "Doc-first, FP8 KV"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(obj, path):
    # Own writer: allow_nan=False makes a NaN fail here instead of breaking the browser import.
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)
    return path.stat().st_size


def provenance(names, paths, data, ri):
    return {
        "exported_utc": ri["timestamp_utc"],
        "export_git_commit": ri["git_commit"],
        "export_git_dirty": ri["git_dirty"],
        "sources": [
            {
                "path": str(paths[n].relative_to(REPO_ROOT)),
                "sha256": sha256(paths[n]),
                "git_commit": data[n].get("run_info", {}).get("git_commit"),
                "timestamp_utc": data[n].get("run_info", {}).get("timestamp_utc"),
            }
            for n in names
        ],
    }


def metrics(mean):
    return {"ndcg10": mean["ndcg@10"], "mrr10": mean["mrr@10"]}


def comparison(c):
    out = {
        "d_ndcg10": c["delta_ndcg@10"],
        "ndcg10_ci": c["ndcg@10_ci95"],
        "d_mrr10": c["delta_mrr@10"],
        "wlt_ndcg10": c["wlt_ndcg@10"],
    }
    if "mrr@10_ci95" in c:
        out["mrr10_ci"] = c["mrr@10_ci95"]
    return out


def export_overview(lora, datasets):
    systems = []
    for key, label, note in SYSTEMS:
        systems.append({"id": key, "label": label, "note": note,
                        "by_ds": {ds: metrics(lora["datasets"][ds][key]["mean"]) for ds in datasets}})
    gate, vs_orig = {}, {}
    for ds in datasets:
        g = lora["datasets"][ds]["gate"]
        gate[ds] = {**comparison(g), "threshold": g["threshold"], "pass": bool(g["pass"])}
        vs_orig[ds] = comparison(lora["datasets"][ds]["dfB_vs_orig_qf"])
    return {
        "datasets": [{"id": ds, "label": DS_LABEL[ds],
                      "num_queries": lora["datasets"][ds]["bm25"]["num_queries"]} for ds in datasets],
        "systems": systems,
        "order_gate": gate,  # doc-first LoRA minus query-first LoRA control
        "dfB_vs_orig_qf": vs_orig,
    }


def export_quant(quant, storage, datasets):
    formats = []
    for fmt in FORMATS:
        label, family, note = FORMAT_INFO[fmt]
        by_ds = {}
        for ds in datasets:
            qd, sd = quant["datasets"][ds], storage["datasets"][ds]["formats"][fmt]
            row = {
                "mib_per_doc": sd["file_mib_per_doc_mean"],
                "bytes_per_doc": sd["file_bytes_per_doc_mean"],
                "ratio_vs_bf16": sd["ratio_vs_bf16_file"],
                "ratio_vs_naive_bf16": sd["ratio_vs_naive_bf16"],
                "corpus_gib": sd["corpus_total_gib"],
            }
            if fmt == "bf16":
                row.update(metrics(qd["baseline"]["mean"]))
                row.update({"d_ndcg10": 0.0, "ndcg10_ci": [0.0, 0.0], "d_mrr10": 0.0, "verdict": "baseline"})
            else:
                f = qd["formats"][fmt]
                assert f["verdict"] in VERDICT_LABEL, f["verdict"]
                row.update(metrics(f["mean"]))
                row.update(comparison(f["vs_bf16"]))
                row["verdict"] = f["verdict"]
                rec = f.get("recon")
                if rec:
                    row["k_rel"], row["v_rel"] = rec["k_rel"], rec["v_rel"]
            by_ds[ds] = row
        formats.append({"id": fmt, "label": label, "family": family, "note": note,
                        "headline": fmt == HEADLINE_FORMAT, "by_ds": by_ds})
    return {
        "margin": {ds: quant["datasets"][ds]["margin"] for ds in datasets},
        "num_queries": {ds: quant["datasets"][ds]["num_queries"] for ds in datasets},
        "verdict_labels": VERDICT_LABEL,
        "formats": formats,
    }


def export_storage(storage, datasets, layers=28, heads=8, head_dim=128):
    sd = {ds: storage["datasets"][ds] for ds in datasets}
    shared_len = sd[datasets[0]]["shared_len"]
    shared_bytes = sd[datasets[0]]["shared_file_bytes"]
    for ds in datasets:  # one model, so the shared prefix file is the same for every corpus
        assert sd[ds]["shared_len"] == shared_len and sd[ds]["shared_file_bytes"] == shared_bytes, ds

    # Exact packed size model: bytes(n) = a + b * n + c * ceil(n / GROUP).
    # The ceil term captures INT4 KIVI's per-32-token key scale groups; c is ~0 for other formats.
    ns = sorted(set(range(8, 513, 8)) | {1, 73, 241, 257, 300, 511})
    x = np.asarray(ns, dtype=np.float64)
    g = np.ceil(x / GROUP)
    A = np.stack([np.ones_like(x), x, g], axis=1)
    fits = []
    for fmt in FORMATS:
        y = np.asarray([predicted_file_bytes(fmt, n, shared_len) for n in ns], dtype=np.float64)
        (a, b, c), *_ = np.linalg.lstsq(A, y, rcond=None)
        rel = np.abs(A @ np.array([a, b, c]) - y) / y
        fits.append({"id": fmt, "label": FORMAT_INFO[fmt][0], "a_bytes": float(a),
                     "b_bytes_per_token": float(b), "c_bytes_per_group": float(c),
                     "fit_max_rel_err": float(rel.max()),
                     "fit_max_rel_err_n_ge_64": float(rel[x >= 64].max())})

    def size(f, t):
        return f["a_bytes"] + f["b_bytes_per_token"] * t + f["c_bytes_per_group"] * np.ceil(t / GROUP)

    fit = {f["id"]: f for f in fits}
    presets = []
    for ds in datasets:
        n, t = sd[ds]["num_docs"], sd[ds]["doc_tokens"]["mean"]
        actual = {fmt: sd[ds]["formats"][fmt]["corpus_total_bytes"] for fmt in FORMATS}
        est = {fmt: float(shared_bytes + n * size(fit[fmt], t)) for fmt in FORMATS}
        presets.append({
            "id": ds, "label": DS_LABEL[ds], "num_docs": n, "mean_doc_tokens": t,
            "doc_tokens_min": sd[ds]["doc_tokens"]["min"], "doc_tokens_max": sd[ds]["doc_tokens"]["max"],
            "corpus_total_bytes": actual, "fit_estimate_bytes": est,
            "fit_rel_err": {fmt: est[fmt] / actual[fmt] - 1 for fmt in FORMATS},
            "naive_bf16_bytes_per_doc": sd[ds]["naive_bf16_bytes_per_doc"],
        })
    return {
        "shared_len": shared_len,
        "shared_file_bytes": shared_bytes,
        "group": GROUP,
        "size_model": "bytes_per_doc = a + b * tokens + c * ceil(tokens / group)",
        "bf16_bytes_per_token_theory": layers * heads * head_dim * 2 * 2,  # K and V, 2 bytes each
        "fit_grid": {"min": ns[0], "max": ns[-1], "points": len(ns)},
        "formats": fits,
        "presets": presets,
    }


def export_trace3(vllm, trace_stats, datasets):
    cap = vllm["capacity"]
    blocks = {cap[s]["block_size"] for s, _ in SETTINGS}
    assert len(blocks) == 1, blocks
    rows = {(r["setting"], r["dataset"], r["trace"]): r for r in vllm["simulation_vs_measured"]}
    out = {}
    for ds in datasets:
        rep = vllm["trace3_repeats"][ds]
        t3 = trace_stats["datasets"][ds]["traces"]["t3"]["repeats"]
        by_setting = {}
        for s, _ in SETTINGS:
            h = vllm["trace3_headline"][ds][s]
            r = rows[(s, ds, "t3")]
            unb = r["sim_unbounded"]["hit_frac"]
            by_setting[s] = {
                "hit_frac": h["hit_frac"],  # measured, cold, c=16
                "doc_tokens_reused_frac": h["doc_tokens_reused_frac"],
                "sim_cold_hit_frac": r["sim_cold"]["hit_frac"],
                "sim_abs_err_c1_cold": r["abs_err_c1"]["cold"],
                "sim_unbounded_hit_frac": unb,
                "sim_unbounded_doc_frac": r["sim_unbounded"]["doc_tokens_reused_frac"],
                "realized_vs_unbounded": h["hit_frac"] / unb if unb > 0 else None,
                "ndcg10": h["ndcg@10"],
                "c1_p50_ms": h["c1_p50_ms"],
                "c16_req_per_s": h["c16_req_per_s"],
                "c16_p50_ms": h["c16_p50_ms"],
                "c16_p95_ms": h["c16_p95_ms"],
            }
        out[ds] = {
            "label": DS_LABEL[ds],
            "n_requests": t3["n_requests"],
            "n_queries": t3["n_queries"],
            "unique_docs": t3["unique_docs_id"],
            "repeat_frac_id": rep["repeat_frac_id"],
            "repeat_frac_content": rep["repeat_frac_content"],
            "reuse_distance_p50": rep["reuse_distance_requests"]["p50"],
            "reuse_distance_p90": rep["reuse_distance_requests"]["p90"],
            "occurrences_per_doc": t3["occurrences_per_doc_id"],
            "by_setting": by_setting,
        }
    return {
        "vllm_version": vllm.get("vllm", {}).get("version") if isinstance(vllm.get("vllm"), dict) else vllm.get("vllm"),
        "block_size": blocks.pop(),
        "settings": [{"id": s, "label": lab, "kv_cache_tokens": int(cap[s]["kv_cache_tokens"]),
                      "num_gpu_blocks": int(cap[s]["num_gpu_blocks"])} for s, lab in SETTINGS],
        "datasets": out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "app" / "data"))
    args = ap.parse_args()

    cfg = load_config()
    datasets = list(cfg.datasets)
    assert set(datasets) <= set(DS_LABEL), datasets
    paths = {n: cfg.paths.results_dir / "metrics" / f"{n}.json" for n in SOURCES}
    data = {n: load_json(paths[n]) for n in SOURCES}
    ri = run_info()
    out = Path(args.out)

    views = {
        "overview": (["lora"], export_overview(data["lora"], datasets)),
        "quant": (["quant", "storage"], export_quant(data["quant"], data["storage"], datasets)),
        "storage": (["storage"], export_storage(data["storage"], datasets)),
        "trace3": (["vllm", "trace_stats"], export_trace3(data["vllm"], data["trace_stats"], datasets)),
    }
    for name, (src, body) in views.items():
        size = write({"provenance": provenance(src, paths, data, ri), **body}, out / f"{name}.json")
        print(f"wrote {out / name}.json ({size:,} bytes)")

    # Spot checks against the README tables.
    q = {f["id"]: f for f in views["quant"][1]["formats"]}
    for ds in datasets:
        k = q[HEADLINE_FORMAT]["by_ds"][ds]
        print(f"[quant] {ds}: int4_kivi nDCG {k['ndcg10']:.4f} d {k['d_ndcg10']:+.4f} "
              f"{k['mib_per_doc']:.2f} MiB/doc {k['ratio_vs_bf16']:.2f}x verdict {k['verdict']}")
    st = views["storage"][1]
    for f in st["formats"]:
        print(f"[storage] {f['id']:<10} a {f['a_bytes']:>8.0f} B  b {f['b_bytes_per_token']:>8.1f} B/tok  "
              f"c {f['c_bytes_per_group']:>9.1f} B/group  fit err max {f['fit_max_rel_err']:.2e}")
    for p in st["presets"]:
        worst = max(abs(v) for v in p["fit_rel_err"].values())
        print(f"[storage] {p['id']}: corpus total from fit vs actual, worst rel err {worst:.2e}")
        assert worst < 1e-2, p["fit_rel_err"]
    t3 = views["trace3"][1]
    for ds in datasets:
        d = t3["datasets"][ds]
        s = d["by_setting"]
        print(f"[trace3] {ds}: repeat {d['repeat_frac_id']:.3f}  hit qf {s['qf']['hit_frac']:.3f} "
              f"df {s['df']['hit_frac']:.3f} fp8 {s['df_fp8']['hit_frac']:.3f}  "
              f"df unbounded {s['df']['sim_unbounded_hit_frac']:.3f}")


if __name__ == "__main__":
    main()