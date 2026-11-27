# scripts/09a_importance.py
"""Phase 6.2: query-independent token importance from eager attention.

For every doc in the rerank set, run its cached prefix (system + instruction +
seam doc) once through the merged doc-first LoRA with eager attention, average
attention over heads and layers, and compute importance (a) and (b) with
marginkv.kv.prune.importance_from_attn. The sweep (6.3) keeps sdpa; this model
is only used here.

Checks on the first --check-docs docs per dataset:
  eager vs sdpa KV (MPS bf16): numeric drift of the eager model, diagnostic.
  importance stability: MPS bf16 vs CPU fp32 eager importance (rank correlation,
  kept-set Jaccard per retention); flag if mean Jaccard at 0.5 < 0.8.

Outputs:
  artifacts/prune/importance_{ds}.pt       (gitignored; imp_a / imp_b per doc, float64)
  results/metrics/prune_importance.json    (diagnostics)
  --smoke N writes under artifacts/smoke/ only.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from marginkv.config import load_config
from marginkv.utils.io import ensure_dir, load_json, load_pt, run_info, save_json, save_pt
from marginkv.utils.seed import set_seed
from marginkv.utils.device import get_device, get_dtype
from marginkv.utils.timing import Timer
from marginkv.kv import prune as P
from marginkv.kv.cache_ops import build_doc_kvs, shared_prefix_len, split_doc_first
from marginkv.models import qwen3_reranker as qr
from marginkv.train.data import content_sha256, sha256_file

PHASE = "6.2"
SETTINGS = {
    "attn_implementation": "eager",
    "average": "mean over 16 query heads and 28 layers, fp32",
    "rows": "doc rows only, raw probabilities (not renormalized over doc keys)",
    "importance_a": "sum_{j>i} A(j,i) / (T-i-1)",
    "importance_b": "sum_{j>i} (shared_len+j+1) A(j,i) / (T-i-1)  (absolute position + 1)",
    "mandatory": {"first": P.FIRST_KEEP, "reserve": P.RESERVE},
    "boundary_first": P.BOUNDARY_FIRST,
    "retentions": list(P.RETENTIONS),
    "batch": 1,
}
ROWSUM_FLAG = 0.05
EAGER_SDPA_FLAG = 0.05
STABILITY_FLAG = 0.8  # mean kept-set Jaccard at retention 0.5, MPS bf16 vs CPU fp32


def load_eager(path, device, dtype):
    m = AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype, attn_implementation="eager")
    m.to(device).eval()
    impl = getattr(m.config, "_attn_implementation", None)
    if impl != "eager":
        raise SystemExit(f"attn implementation is {impl!r}, expected eager")
    return m


def verify_model(path):
    meta = load_json(path / "marginkv_merge.json")
    for fn, h in meta["files_sha256"].items():
        got = sha256_file(path / fn)
        if got != h:
            raise SystemExit(f"sha256 mismatch for {fn}: {got} != {h}")
    return meta["files_sha256"]


@torch.no_grad()
def doc_attention(model, ids, device):
    x = torch.tensor([ids], device=device)
    pos = torch.arange(len(ids), device=device)[None]
    out = model.model(input_ids=x, position_ids=pos, output_attentions=True, use_cache=True)
    att = out.attentions
    if att is None or len(att) != model.config.num_hidden_layers:
        raise SystemExit("no attentions returned: model is not running eager attention")
    S = len(ids)
    avg = torch.zeros(S, S, device=device, dtype=torch.float32)
    for a in att:
        avg += a[0].float().mean(0)  # [16, S, S] -> [S, S]
    avg /= len(att)
    return avg, out.past_key_values


def spearman(x, y):
    rx = x.argsort().argsort().double()
    ry = y.argsort().argsort().double()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = (rx.norm() * ry.norm()).item()
    return float((rx @ ry).item() / d) if d > 0 else None


def jaccard(a, b):
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def free_idx(T):
    mand = set(P.mandatory_idx(T))
    return mand, [i for i in range(T) if i not in mand]


def kept_free(method, T, r, mand, importance=None):
    return set(P.select_keep(method, T, r, importance=importance)[0].tolist()) - mand


def doc_stats(avg, s, T, ia, ib):
    rows = avg[s:]
    st = {
        "T": T,
        "prefix_mass": rows[:, :s].sum(1).mean().item(),
        "first_token_mass": rows[:, 0].mean().item(),
        "rowsum_dev": (avg.sum(1) - 1).abs().max().item(),
    }
    mand, free = free_idx(T)
    if len(free) >= 3:
        pos = torch.tensor(free, dtype=torch.float64)
        st["spearman_a"] = spearman(ia[free], pos)
        st["spearman_b"] = spearman(ib[free], pos)
    ov = {}
    for r in P.RETENTIONS:
        ka = kept_free("attn_a", T, r, mand, ia)
        kb = kept_free("attn_b", T, r, mand, ib)
        kd = kept_free("boundary", T, r, mand)
        ov[str(r)] = {
            "jaccard_a_b": jaccard(ka, kb),
            "jaccard_a_boundary": jaccard(ka, kd),
            "jaccard_b_boundary": jaccard(kb, kd),
            "early_half_a": (sum(i < T / 2 for i in ka) / len(ka)) if ka else None,
            "early_half_b": (sum(i < T / 2 for i in kb) / len(kb)) if kb else None,
        }
    st["overlap"] = ov
    return st


def dist(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "min": float(a.min()), "p10": float(np.percentile(a, 10)),
            "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90))}


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def eager_sdpa_check(eager, sdpa, prefixes, device, s):
    k_rel, v_rel = [], []
    for ids in prefixes:
        _, pkv = doc_attention(eager, ids, device)
        ref = build_doc_kvs(sdpa, [ids], device, s)[0]
        k_rel.append([rel(pkv.layers[l].keys[0], ref.keys[l]) for l in range(ref.num_layers)])
        v_rel.append([rel(pkv.layers[l].values[0], ref.values[l]) for l in range(ref.num_layers)])
    k, v = np.asarray(k_rel), np.asarray(v_rel)
    return {"n_docs": len(prefixes), "k_rel_layer0_max": float(k[:, 0].max()), "v_rel_layer0_max": float(v[:, 0].max()),
            "k_rel_max": float(k.max()), "v_rel_max": float(v.max()),
            "k_rel_mean": float(k.mean()), "v_rel_mean": float(v.mean()),
            "k_worst_layer": int(k.max(0).argmax()), "v_worst_layer": int(v.max(0).argmax()),
            "flag": bool(max(k.max(), v.max()) > EAGER_SDPA_FLAG)}


def importance_stability(eager, cpu_model, prefixes, device, s):
    out = {"n_docs": len(prefixes)}
    per = {"a": {"spearman": [], **{str(r): [] for r in P.RETENTIONS}},
           "b": {"spearman": [], **{str(r): [] for r in P.RETENTIONS}}}
    for ids in prefixes:
        avg_m, _ = doc_attention(eager, ids, device)
        avg_c, _ = doc_attention(cpu_model, ids, torch.device("cpu"))
        T = len(ids) - s
        mand, free = free_idx(T)
        for name, w in (("a", False), ("b", True)):
            im = P.importance_from_attn(avg_m, s, weighted=w)
            ic = P.importance_from_attn(avg_c, s, weighted=w)
            if len(free) >= 3:
                per[name]["spearman"].append(spearman(im[free], ic[free]))
            for r in P.RETENTIONS:
                per[name][str(r)].append(jaccard(kept_free(f"attn_{name}", T, r, mand, im),
                                                 kept_free(f"attn_{name}", T, r, mand, ic)))
    for name in ("a", "b"):
        out[name] = {k: dist(v) for k, v in per[name].items()}
    j05 = [out[n]["0.5"].get("mean") for n in ("a", "b")]
    out["flag"] = bool(any(x is not None and x < STABILITY_FLAG for x in j05))
    return out


def run_dataset(ds, cfg, args, eager, get_sdpa, get_cpu, device, model_sha, out_dir, part_dir):
    art = load_pt(cfg.paths.artifacts_dir / "tokenized" / f"{ds}_qwen3.pt")
    dfart = load_pt(cfg.paths.artifacts_dir / "tokenized" / f"{ds}_qwen3_df.pt")
    assert dfart["base_content_sha256"] == art["content_sha256"]
    pieces = {**art["pieces"], **dfart["seam_pieces"]}
    s = shared_prefix_len(pieces)
    docs = list(dfart["doc_order"])
    if args.smoke:
        docs = docs[:args.smoke]
    key = content_sha256({"settings": SETTINGS, "art": art["content_sha256"], "dfart": dfart["content_sha256"],
                          "model": model_sha, "docs": docs, "check_docs": args.check_docs})

    final = out_dir / f"importance_{ds}.pt"
    if final.exists() and not args.force:
        old = load_pt(final)
        if old["key"] == key:
            print(f"[{ds}] reusing {final}")
            return old
        raise SystemExit(f"[{ds}] {final} exists with different settings; use --force")

    part = part_dir / f"importance_{ds}.partial.pt"
    state = load_pt(part) if part.exists() else None
    if state is None or state["key"] != key or args.force:
        state = {"key": key, "done": 0, "imp_a": {}, "imp_b": {}, "stats": {}, "ms": [],
                 "check": None, "stability": None}

    if args.check_docs > 0 and (state["check"] is None or state["stability"] is None):
        pre = [split_doc_first(pieces, dfart["doc_seam"][d], [], variant="B")[0] for d in docs[:args.check_docs]]
        if state["check"] is None:
            state["check"] = eager_sdpa_check(eager, get_sdpa(), pre, device, s)
            c = state["check"]
            print(f"[{ds}] eager vs sdpa KV rel: layer0 K {c['k_rel_layer0_max']:.1e}, max K {c['k_rel_max']:.2e} "
                  f"(L{c['k_worst_layer']}) V {c['v_rel_max']:.2e} (L{c['v_worst_layer']}), "
                  f"mean K {c['k_rel_mean']:.2e} V {c['v_rel_mean']:.2e}{'  FLAG' if c['flag'] else ''}")
        if state["stability"] is None:
            state["stability"] = importance_stability(eager, get_cpu(), pre, device, s)
            st = state["stability"]
            print(f"[{ds}] importance MPS bf16 vs CPU fp32: spearman a {st['a']['spearman'].get('mean', float('nan')):.3f} "
                  f"b {st['b']['spearman'].get('mean', float('nan')):.3f}; kept-set jaccard a/b @0.25 "
                  f"{st['a']['0.25'].get('mean', float('nan')):.3f}/{st['b']['0.25'].get('mean', float('nan')):.3f} "
                  f"@0.5 {st['a']['0.5'].get('mean', float('nan')):.3f}/{st['b']['0.5'].get('mean', float('nan')):.3f}"
                  f"{'  FLAG' if st['flag'] else ''}")
        save_pt(state, part)

    total = len(docs)
    for start in range(state["done"], total, args.chunk):
        for d in docs[start:start + args.chunk]:
            ids, _ = split_doc_first(pieces, dfart["doc_seam"][d], [], variant="B")
            T = len(ids) - s
            assert T == len(dfart["doc_seam"][d]) and len(ids) == dfart["cache_len_seam"][d]
            with Timer(device) as t:
                avg, _ = doc_attention(eager, ids, device)
            ia = P.importance_from_attn(avg, s, weighted=False)
            ib = P.importance_from_attn(avg, s, weighted=True)
            if not (torch.isfinite(ia[:-1]).all() and torch.isfinite(ib[:-1]).all()):
                raise SystemExit(f"[{ds}] non-finite importance for doc {d}")
            state["imp_a"][d] = ia
            state["imp_b"][d] = ib
            state["stats"][d] = doc_stats(avg.cpu(), s, T, ia, ib)
            state["ms"].append(t.ms)
        state["done"] = min(start + args.chunk, total)
        save_pt(state, part)
        msd = float(np.mean(state["ms"][-args.chunk:]))
        eta = (total - state["done"]) * msd / 1000 / 60
        print(f"[{ds}] {state['done']}/{total} docs, {msd:.0f} ms/doc (last chunk), ETA {eta:.1f} min")

    out = {
        "dataset": ds, "key": key, "phase": PHASE, "settings": SETTINGS, "model_files_sha256": model_sha,
        "base_content_sha256": art["content_sha256"], "df_content_sha256": dfart["content_sha256"],
        "shared_len": s, "doc_order": docs, "doc_len": {d: state["stats"][d]["T"] for d in docs},
        "imp_a": state["imp_a"], "imp_b": state["imp_b"], "stats": state["stats"], "ms": state["ms"],
        "check": state["check"], "stability": state["stability"], "run_info": run_info(),
    }
    save_pt(out, final)
    part.unlink(missing_ok=True)
    print(f"[{ds}] saved {final}")
    return out


def summarize(res):
    st = list(res["stats"].values())
    ms = res["ms"]
    ov = {}
    for r in P.RETENTIONS:
        k = str(r)
        ov[k] = {f: dist([x["overlap"][k][f] for x in st]) for f in
                 ("jaccard_a_b", "jaccard_a_boundary", "jaccard_b_boundary", "early_half_a", "early_half_b")}
    rowsum_max = max(x["rowsum_dev"] for x in st)
    return {
        "num_docs": len(st),
        "shared_len": res["shared_len"],
        "doc_tokens": dist([x["T"] for x in st]),
        "timing": {"ms_per_doc_mean": float(np.mean(ms)), "ms_per_doc_p50": float(np.median(ms)),
                   "total_min": float(np.sum(ms) / 60000)},
        "rowsum_dev_max": rowsum_max,
        "rowsum_flag": bool(rowsum_max > ROWSUM_FLAG),
        "eager_vs_sdpa": res["check"],
        "importance_stability": res["stability"],
        "sink": {"prefix_mass": dist([x["prefix_mass"] for x in st]),
                 "first_token_mass": dist([x["first_token_mass"] for x in st])},
        "position_spearman": {"a": dist([x.get("spearman_a") for x in st]),
                              "b": dist([x.get("spearman_b") for x in st])},
        "overlap": ov,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--check-docs", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)
    device = get_device(args.device if args.device != "auto" else cfg.runtime.device)
    dtype = get_dtype(cfg.runtime.dtype)
    datasets = args.datasets or list(cfg.datasets)

    if args.smoke:
        out_dir = ensure_dir(cfg.paths.artifacts_dir / "smoke" / "prune")
        part_dir = out_dir
        metrics_path = cfg.paths.artifacts_dir / "smoke" / "prune_importance.json"
    else:
        out_dir = ensure_dir(cfg.paths.artifacts_dir / "prune")
        part_dir = ensure_dir(cfg.paths.artifacts_dir / "prune" / "partial")
        metrics_path = cfg.paths.results_dir / "metrics" / "prune_importance.json"

    mdir = cfg.paths.artifacts_dir / "lora" / "df" / "merged"
    model_sha = verify_model(mdir)
    t0 = time.time()
    eager = load_eager(mdir, device, dtype)
    print(f"eager model loaded in {time.time() - t0:.1f}s on {device} {dtype}")

    holder = {}

    def get_sdpa():
        if "sdpa" not in holder:
            holder["sdpa"] = qr.load_model(str(mdir), None, device, dtype)
        return holder["sdpa"]

    def get_cpu():
        if "cpu" not in holder:
            torch.set_num_threads(4)
            holder["cpu"] = load_eager(mdir, torch.device("cpu"), torch.float32)
        return holder["cpu"]

    summary = {}
    for ds in datasets:
        res = run_dataset(ds, cfg, args, eager, get_sdpa, get_cpu, device, model_sha, out_dir, part_dir)
        summary[ds] = summarize(res)
        sm = summary[ds]
        print(f"[{ds}] prefix mass p50 {sm['sink']['prefix_mass'].get('p50', float('nan')):.3f}, "
              f"first-token mass p50 {sm['sink']['first_token_mass'].get('p50', float('nan')):.3f}, "
              f"spearman(pos) a {sm['position_spearman']['a'].get('mean', float('nan')):+.3f} "
              f"b {sm['position_spearman']['b'].get('mean', float('nan')):+.3f}, "
              f"jaccard a-b @0.5 {sm['overlap']['0.5']['jaccard_a_b'].get('mean', float('nan')):.3f}, "
              f"a-boundary @0.5 {sm['overlap']['0.5']['jaccard_a_boundary'].get('mean', float('nan')):.3f}, "
              f"rowsum dev {sm['rowsum_dev_max']:.1e}")
    holder.clear()

    save_json({"run_info": run_info(), "phase": PHASE, "settings": SETTINGS, "model_files_sha256": model_sha,
               "smoke_docs": args.smoke or None, "datasets": summary}, metrics_path)
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()