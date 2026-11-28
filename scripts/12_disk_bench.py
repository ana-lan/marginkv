# scripts/12_disk_bench.py
"""Phase 8b.1: packed document KV on disk. Bytes per doc and cold / warm load time to a usable cache on the
run device, split into read / parse / h2d / decode, vs. recomputing the doc KV (prefill).

Per arm (6 Phase 5 formats + 2 Phase 7 combined): build each sampled doc's KV (batch 1), transform on CPU,
pack_doc, fsync, evict from the page cache. Size == predicted_file_bytes (abort otherwise); CPU unpack ==
transform output (abort otherwise); stage-split decode on the run device == transform output (flag otherwise).
Then evict every file, and per doc in one seeded order: page-cache residency (mincore), cold load, warm load.
Files are deleted after each arm unless --keep (all 8 arms at 1,000 docs is about 100 GB).
Rules in RULES are pre-registered (committed before the pod run). Report only, no gates.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import ctypes
import hashlib
import json
import mmap
import random
import shutil
import statistics
import sys
import time

import torch
from safetensors.torch import load as st_load

from marginkv.config import load_config
from marginkv.kv import cache_ops as co
from marginkv.kv import combined as CB
from marginkv.kv import pack
from marginkv.kv import quant as Q
from marginkv.kv import sweep
from marginkv.models import qwen3_reranker as qr
from marginkv.train.data import sha256_file
from marginkv.utils.device import get_device, get_dtype
from marginkv.utils.io import ensure_dir, load_json, load_pt, run_info, save_json
from marginkv.utils.timing import Timer, summarize_ms

PHASE = "8b.1"
MIB = 2 ** 20
STAGES = ("read", "parse", "h2d", "decode")
ARMS = {f: ("none", 1.0, f) for f in Q.FORMATS}
ARMS.update({CB.arm_name("uniform", 0.75, f): ("uniform", 0.75, f) for f in ("fp8_tensor", "int4_kivi")})
EVICT = hasattr(os, "posix_fadvise")
LINUX = sys.platform.startswith("linux")
RESIDENT_FLAG = 0.01

RULES = {
    "phase": PHASE,
    "sample": "per dataset: shuffle the seam artifact's doc_order with random.Random(f'{seed}-disk-{ds}'), take the "
              "first per_ds (500); same docs for every arm; --smoke N takes the first N of the same shuffle",
    "arms": {a: list(v) for a, v in ARMS.items()},
    "build": "doc KV built on the run device one doc at a time (build_doc_kvs, batch 1), moved to CPU; transform on "
             "CPU (quantize_doc, or uniform 75% prune then quantize via kv.combined); pack_doc; fsync",
    "size_check": "file bytes == predicted_file_bytes(fmt, kept, shared_len, next_pos) == size on disk, every file; abort otherwise",
    "exactness": {"cpu": "pack.unpack_doc on CPU == transform output exactly (every checked file); abort otherwise",
                  "device": "stage-split decode on the run device == transform output exactly; mismatch is a flag "
                            "(count and max abs diff recorded), no abort"},
    "cold": "fsync + posix_fadvise(DONTNEED) on each file after writing, again on every file before the timed pass; "
            "page-cache residency measured with mincore just before each cold read (Linux)",
    "read_order": "one seeded shuffle of all sampled docs, random.Random(f'{seed}-disk-order'), same order for every arm",
    "per_doc": "cold load, then immediately a warm load of the same file (paired)",
    "stages": "read = whole-file read; parse = safetensors header + tensors on CPU; h2d = each tensor .to(device) from "
              "pageable memory, synced; decode = dequantize + concat with the on-device shared prefix, synced; total = sum",
    "warmup": "the first `warmup` docs of the read order run cold + warm untimed, are evicted again, then timed in order",
    "shared_prefix": "canonical shared prefix built once per dataset, kept on the device, not counted per doc",
    "recompute": "write-phase build time per doc (prefill of system + instruction + doc, batch 1, synced), first "
                 "`warmup` builds of each arm dropped, pooled over arms",
    "report": "per arm, cold and warm: p50 / p95 / mean per stage and total; MB/s p50 (file bytes / total); "
              "cold total p50 / recompute p50",
    "flags": {"eviction": f"mean residency before cold reads > {RESIDENT_FLAG} (unverified where mincore is unavailable)",
              "device_exact": "any device decode mismatch"},
    "gates": None,
}
RULES_SHA = hashlib.sha256(json.dumps(RULES, sort_keys=True).encode()).hexdigest()


# ---------- page cache ----------

def sync_evict(path, sync=True):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        if sync:
            os.fsync(fd)
        if EVICT:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


_libc = None


def resident_frac(path):
    """Fraction of the file's pages in the page cache (mincore on an untouched private mapping). None off Linux."""
    global _libc
    if not LINUX:
        return None
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
        _libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
        _libc.mincore.restype = ctypes.c_int
    size = os.path.getsize(path)
    npages = -(-size // mmap.PAGESIZE)
    vec = (ctypes.c_ubyte * npages)()
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_COPY)
        buf = (ctypes.c_char * size).from_buffer(mm)
        try:
            rc = _libc.mincore(ctypes.addressof(buf), size, vec)
        finally:
            del buf
            mm.close()
    if rc != 0:
        raise OSError(ctypes.get_errno(), "mincore failed")
    return sum(b & 1 for b in bytes(vec)) / npages


# ---------- load stages ----------

def read_file(path):
    with open(path, "rb", buffering=0) as f:
        return f.readall()


def parse_packed(data):
    n = int.from_bytes(data[:8], "little")
    meta = json.loads(data[8:8 + n]).get("__metadata__", {})
    if meta.get("version") != pack.VERSION:
        raise ValueError(f"unknown packed version {meta.get('version')!r}")
    return st_load(data), meta


def decode_packed(t, meta, shared_k, shared_v):
    """Same decode as pack.unpack_doc, on tensors already on the device; shared_* already on the device."""
    fmt, dt, L = meta["format"], getattr(torch, meta["dtype"]), int(meta["layers"])
    out = {}
    for role, shared in (("k", shared_k), ("v", shared_v)):
        if fmt == "bf16":
            docs = list(t[role].unbind(0))
        else:
            kind, _ = Q.SPEC[fmt][role]
            codes = t[f"{role}_codes"]
            if kind == "fp8":
                codes = codes.view(torch.float8_e4m3fn)
            elif kind == "asym":
                codes = pack.unpack_nibbles(codes)
            docs = []
            for l in range(L):
                enc = {"codes": codes[l], "scale": t[f"{role}_scale"][l]}
                if kind == "asym":
                    enc["min"] = t[f"{role}_min"][l]
                docs.append(Q.decode(enc, fmt, role, dt))
        out[role] = [torch.cat([sh, d], dim=1) for sh, d in zip(shared, docs)]
    return co.DocKV(out["k"], out["v"], int(meta["next_pos"]), int(meta["shared_len"]))


def load_once(path, shared, device):
    t0 = time.perf_counter_ns()
    data = read_file(path)
    t1 = time.perf_counter_ns()
    tensors, meta = parse_packed(data)
    t2 = time.perf_counter_ns()
    with Timer(device) as th:
        td = {k: v.to(device) for k, v in tensors.items()}
    with Timer(device) as tdc:
        kv = decode_packed(td, meta, *shared)
    ms = {"read": (t1 - t0) / 1e6, "parse": (t2 - t1) / 1e6, "h2d": float(th.ms), "decode": float(tdc.ms)}
    ms["total"] = sum(ms[s] for s in STAGES)
    return ms, kv


def kv_diff(a, b):
    if (a.next_pos, a.shared_len, a.n_cached, a.num_layers) != (b.next_pos, b.shared_len, b.n_cached, b.num_layers):
        return float("inf")
    m = 0.0
    for l in range(a.num_layers):
        for x, y in ((a.keys[l], b.keys[l]), (a.values[l], b.values[l])):
            m = max(m, float((x.float() - y.float()).abs().max()))
    return m


# ---------- summaries ----------

def stats(xs):
    d = summarize_ms(list(xs))
    return {k: (int(v) if k == "n" else (None if v is None else float(v))) for k, v in d.items()}


def summarize_arm(rows, build_ms, warmup, datasets):
    out = {"n": len(rows),
           "bytes_per_doc_mean": float(statistics.fmean(r["file_bytes"] for r in rows)),
           "doc_tokens_mean": float(statistics.fmean(r["doc_tokens"] for r in rows)),
           "kept_mean": float(statistics.fmean(r["kept"] for r in rows))}
    out["mib_per_doc_mean"] = out["bytes_per_doc_mean"] / MIB
    for mode in ("cold", "warm"):
        out[mode] = {s: stats(r[mode][s] for r in rows) for s in (*STAGES, "total")}
        out[f"{mode}_mb_per_s_p50"] = float(statistics.median(r["file_bytes"] / (r[mode]["total"] * 1e3) for r in rows))
    res = [r["resident_before"] for r in rows if r["resident_before"] is not None]
    out["resident_before_mean"] = float(statistics.fmean(res)) if res else None
    out["resident_before_max"] = float(max(res)) if res else None
    out["by_ds"] = {ds: {"n": sum(r["ds"] == ds for r in rows),
                         "cold_total_p50": float(statistics.median(r["cold"]["total"] for r in rows if r["ds"] == ds)),
                         "warm_total_p50": float(statistics.median(r["warm"]["total"] for r in rows if r["ds"] == ds))}
                    for ds in datasets if any(r["ds"] == ds for r in rows)}
    out["build_ms"] = stats(build_ms[warmup:] or build_ms)
    return out


def verify_model(model_dir):
    shas = load_json(model_dir / "marginkv_merge.json")["files_sha256"]
    for fname, want in shas.items():
        if sha256_file(model_dir / fname) != want:
            raise SystemExit(f"sha mismatch: {fname}")
    return shas


# ---------- one arm ----------

def run_arm(arm, model, device, plans, docs, order, shared, args, root):
    method, retention, fmt = ARMS[arm]
    tf = (lambda kv, d: Q.quantize_doc(kv, fmt)) if method == "none" else CB.make_combined(method, retention, fmt)
    adir = root / arm
    if adir.exists():
        shutil.rmtree(adir)

    upper = sum(pack.predicted_file_bytes(fmt, plans[ds]["dfart"]["cache_len_seam"][d] - plans[ds]["s"], plans[ds]["s"])
                for ds, d in docs)
    free = shutil.disk_usage(ensure_dir(root)).free
    if free < 1.1 * upper:
        raise SystemExit(f"{arm}: need about {upper / 2**30:.1f} GiB, only {free / 2**30:.1f} GiB free")

    n_check = len(docs) if args.check < 0 else min(args.check, len(docs))
    checks = {"cpu_checked": 0, "cpu_exact": 0, "device_checked": 0, "device_exact": 0, "device_max_abs": 0.0}
    build_ms, pack_ms, info = [], [], {}
    t_start = time.time()
    for i, (ds, d) in enumerate(docs):
        P = plans[ds]
        s = P["s"]
        prefix, _ = co.split_doc_first(P["pieces"], P["dfart"]["doc_seam"][d], [], variant="B")
        with Timer(device) as tb:
            kv_dev = co.build_doc_kvs(model, [prefix], device, s)[0]
        build_ms.append(float(tb.ms))
        kv = kv_dev.to("cpu")
        del kv_dev
        t0 = time.perf_counter_ns()
        ref = tf(kv, d)
        path = adir / ds / f"{d}.safetensors"
        pk = pack.pack_doc(ref, fmt, path)
        sync_evict(path)
        pack_ms.append((time.perf_counter_ns() - t0) / 1e6)
        kept = ref.n_cached - s
        pred = pack.predicted_file_bytes(fmt, kept, s, next_pos=ref.next_pos)
        on_disk = path.stat().st_size
        if not (pk["file_bytes"] == pred == on_disk):
            raise SystemExit(f"{arm} {ds}/{d}: size {pk['file_bytes']} / predicted {pred} / on disk {on_disk}")
        info[(ds, d)] = {"path": path, "doc_tokens": ref.next_pos - s, "kept": kept, "file_bytes": on_disk}

        if i < n_check:
            sk, sv = [k[:, :s] for k in ref.keys], [v[:, :s] for v in ref.values]
            dc = kv_diff(pack.unpack_doc(path, sk, sv), ref)
            checks["cpu_checked"] += 1
            if dc != 0:
                raise SystemExit(f"{arm} {ds}/{d}: CPU unpack differs from the transform output (max {dc})")
            checks["cpu_exact"] += 1
            _, out = load_once(path, ([x.to(device) for x in sk], [x.to(device) for x in sv]), device)
            dd = kv_diff(out.to("cpu"), ref)
            del out
            checks["device_checked"] += 1
            checks["device_exact"] += int(dd == 0)
            checks["device_max_abs"] = max(checks["device_max_abs"], dd)
            sync_evict(path, sync=False)  # the check read it back into the page cache
        if (i + 1) % 100 == 0 or i + 1 == len(docs):
            print(f"  [{arm}] wrote {i + 1}/{len(docs)}  build p50 {statistics.median(build_ms):.1f} ms  "
                  f"device exact {checks['device_exact']}/{checks['device_checked']}  {time.time() - t_start:.0f} s")

    for v in info.values():
        sync_evict(v["path"], sync=False)
    W = min(args.warmup, len(order))
    for key in order[:W]:
        for _ in range(2):
            _, kv = load_once(info[key]["path"], shared[key[0]], device)
            del kv
        sync_evict(info[key]["path"], sync=False)

    rows = []
    for key in order:
        ds, d = key
        v = info[key]
        res = resident_frac(v["path"])
        cold, kv = load_once(v["path"], shared[ds], device)
        del kv
        warm, kv = load_once(v["path"], shared[ds], device)
        del kv
        rows.append({"ds": ds, "doc": d, "doc_tokens": v["doc_tokens"], "kept": v["kept"],
                     "file_bytes": v["file_bytes"], "resident_before": res, "cold": cold, "warm": warm})

    if not args.keep:
        shutil.rmtree(adir)
    summ = summarize_arm(rows, build_ms, W, list(plans))
    return {"checks": checks, "build_ms": build_ms, "pack_ms": pack_ms, "rows": rows, "summary": summ}


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-ds", type=int, default=500)
    ap.add_argument("--smoke", type=int, default=0, help="docs per dataset; outputs under artifacts/smoke/")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--check", type=int, default=-1, help="exactness-checked docs per arm (-1 = all)")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    torch.set_grad_enabled(False)
    A, R = cfg.paths.artifacts_dir, cfg.paths.results_dir
    per_ds = args.smoke or args.per_ds
    if args.smoke:
        out_dir, metrics_path, root = A / "smoke" / "disk", A / "smoke" / "disk" / "disk.json", A / "smoke" / "disk_bench"
    else:
        out_dir, metrics_path, root = R / "disk", R / "metrics" / "disk.json", A / "disk_bench"
    ensure_dir(out_dir)
    device, dtype = get_device(args.device or cfg.runtime.device), get_dtype(cfg.runtime.dtype)
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    settings = {"per_ds": per_ds, "smoke": bool(args.smoke), "warmup": args.warmup, "check": args.check,
                "device": str(device), "dtype": cfg.runtime.dtype, "gpu": gpu, "platform": sys.platform,
                "eviction": "posix_fadvise" if EVICT else "none", "residency_check": "mincore" if LINUX else "none",
                "ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"), "root": str(root)}

    plans, docs = {}, []
    for ds in cfg.datasets:
        art = load_pt(A / "tokenized" / f"{ds}_qwen3.pt")
        dfart = load_pt(A / "tokenized" / f"{ds}_qwen3_df.pt")
        assert dfart["base_content_sha256"] == art["content_sha256"]
        pieces, _, _ = sweep.sweep_plan(art, dfart)
        ids = list(dfart["doc_order"])
        random.Random(f"{cfg.seed}-disk-{ds}").shuffle(ids)
        plans[ds] = {"pieces": pieces, "dfart": dfart, "s": co.shared_prefix_len(pieces)}
        docs += [(ds, d) for d in ids[:per_ds]]
    order = list(docs)
    random.Random(f"{cfg.seed}-disk-order").shuffle(order)
    settings["n_docs"] = len(docs)

    todo = []
    for arm in args.arms:
        p = out_dir / f"{arm}.json"
        if p.exists() and not args.force:
            old = load_json(p)
            if old["rules_sha256"] != RULES_SHA or old["settings"]["n_docs"] != len(docs):
                raise SystemExit(f"{p} was made with other rules or another sample; use --force")
            print(f"[{arm}] reusing {p}")
        else:
            todo.append(arm)

    if todo:
        model_dir = A / "lora" / "df" / "merged"
        model_sha = verify_model(model_dir)
        model = qr.load_model(str(model_dir), None, device, dtype)
        shared = {}
        for ds, P in plans.items():
            skv = co.build_doc_kvs(model, [P["pieces"]["prefix"] + P["pieces"]["df_head"]], device, P["s"])[0]
            shared[ds] = (skv.keys, skv.values)
        print(f"device {device} {gpu or ''} | eviction {settings['eviction']} | residency {settings['residency_check']} "
              f"| {len(docs)} docs | arms {todo}")
        for arm in todo:
            method, retention, fmt = ARMS[arm]
            res = run_arm(arm, model, device, plans, docs, order, shared, args, root)
            rec = {"run_info": run_info(), "phase": PHASE, "arm": arm, "method": method, "retention": retention,
                   "format": fmt, "rules_sha256": RULES_SHA, "model_files_sha256": model_sha, "settings": settings,
                   "shared_len": plans[cfg.datasets[0]]["s"], **res}
            save_json(rec, out_dir / f"{arm}.json")
            s = res["summary"]
            print(f"[{arm}] {s['mib_per_doc_mean']:6.2f} MiB  cold p50 {s['cold']['total']['p50']:6.1f} "
                  f"p95 {s['cold']['total']['p95']:6.1f} ms  warm p50 {s['warm']['total']['p50']:6.1f} ms  "
                  f"read {s['cold']['read']['p50']:5.1f}  decode {s['cold']['decode']['p50']:5.1f}  "
                  f"{s['cold_mb_per_s_p50']:7.0f} MB/s  resident {s['resident_before_mean']}")

    recs = {a: load_json(out_dir / f"{a}.json") for a in ARMS if (out_dir / f"{a}.json").exists()}
    pooled = [x for r in recs.values() for x in (r["build_ms"][r["settings"]["warmup"]:] or r["build_ms"])]
    recompute = stats(pooled)
    arms_out, flags = {}, {"eviction": [], "eviction_unverified": [], "device_exact": []}
    for a, r in recs.items():
        s, c = r["summary"], r["checks"]
        arms_out[a] = {**s, "checks": c, "method": r["method"], "retention": r["retention"], "format": r["format"],
                       "cold_over_recompute_p50": s["cold"]["total"]["p50"] / recompute["p50"],
                       "warm_over_recompute_p50": s["warm"]["total"]["p50"] / recompute["p50"]}
        if s["resident_before_mean"] is None:
            flags["eviction_unverified"].append(a)
        elif s["resident_before_mean"] > RESIDENT_FLAG:
            flags["eviction"].append(a)
        if c["device_exact"] < c["device_checked"]:
            flags["device_exact"].append(a)
    report = {"run_info": run_info(), "phase": PHASE, "rules": RULES, "rules_sha256": RULES_SHA,
              "settings": settings, "recompute_ms": recompute, "arms": arms_out, "flags": flags}
    save_json(report, metrics_path)

    print(f"\nrecompute (prefill, batch 1) p50 {recompute['p50']:.1f} ms, p95 {recompute['p95']:.1f} ms, n {recompute['n']}")
    for a, v in arms_out.items():
        print(f"{a:24s} {v['mib_per_doc_mean']:6.2f} MiB  cold p50 {v['cold']['total']['p50']:6.1f} ms "
              f"({v['cold_over_recompute_p50']:.2f}x recompute)  warm {v['warm']['total']['p50']:6.1f} ms  "
              f"exact cpu {v['checks']['cpu_exact']}/{v['checks']['cpu_checked']} "
              f"device {v['checks']['device_exact']}/{v['checks']['device_checked']}")
    print(f"flags: {flags}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()