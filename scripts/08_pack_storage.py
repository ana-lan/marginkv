# scripts/08_pack_storage.py
"""Phase 5.5: packed storage, real bytes, bit-exact round trip.

Per dataset:
  1. Build the canonical shared prefix (system + instruction, 59 tokens) once and save it.
  2. Stratified sample of docs (shortest to longest): build each doc KV (MPS bf16, batch 1), then for
     every format pack to safetensors, check file size == predicted, payload == theory_bytes, and
     unpack == quantize_doc bit for bit (CPU; MPS quantize must also equal CPU). Warm-cache load +
     decode time (preview only; cold disk numbers are Phase 8).
  3. Shared-prefix check: each sampled doc's own prefix KV vs the canonical one (bit-identical?), and
     the score shift from pairing the doc's tokens with the canonical prefix (vs the 5.2 noise floor).
  4. Corpus totals for every reranked doc from predicted_file_bytes (exact, verified on the sample).
Sample files are deleted after checking unless --keep. Writes results/metrics/storage.json.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse

import numpy as np
import torch

from marginkv.config import load_config
from marginkv.kv import cache_ops as co
from marginkv.kv import pack as P
from marginkv.kv import quant as Q
from marginkv.kv import sweep
from marginkv.models import qwen3_reranker as qr
from marginkv.train.data import sha256_file
from marginkv.utils.device import get_device, get_dtype
from marginkv.utils.io import ensure_dir, load_json, load_pt, run_info, save_json
from marginkv.utils.seed import set_seed
from marginkv.utils.timing import Timer, summarize_ms

FORMATS = Q.FORMATS
MiB = 1024 ** 2


def verify_model(model_dir):
    shas = load_json(model_dir / "marginkv_merge.json")["files_sha256"]
    for fname, want in shas.items():
        if sha256_file(model_dir / fname) != want:
            raise SystemExit(f"sha mismatch: {fname}")
    return shas


def rel_diff(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=16, help="docs per dataset to pack and check")
    ap.add_argument("--keep", action="store_true", help="keep the packed sample files")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)
    torch.set_grad_enabled(False)
    device, dtype = get_device(cfg.runtime.device), get_dtype(cfg.runtime.dtype)
    A, R = cfg.paths.artifacts_dir, cfg.paths.results_dir
    cpu = torch.device("cpu")

    model_dir = A / "lora" / "df" / "merged"
    shas = verify_model(model_dir)
    model = qr.load_model(str(model_dir), None, device, dtype)
    L, H = model.config.num_hidden_layers, model.config.num_key_value_heads
    D = getattr(model.config, "head_dim", None) or model.config.hidden_size // model.config.num_attention_heads

    report = {"run_info": run_info(), "phase": "5.5", "version": P.VERSION, "model_files_sha256": shas,
              "sample_per_dataset": args.sample, "kept_files": args.keep, "datasets": {}}

    for ds in cfg.datasets:
        art = load_pt(A / "tokenized" / f"{ds}_qwen3.pt")
        dfart = load_pt(A / "tokenized" / f"{ds}_qwen3_df.pt")
        assert dfart["base_content_sha256"] == art["content_sha256"], "seam artifact mismatch"
        pieces, docs, by_doc = sweep.sweep_plan(art, dfart)
        s = co.shared_prefix_len(pieces)
        n_doc = {d: dfart["cache_len_seam"][d] - s for d in docs}
        root = ensure_dir(A / "kv_packed" / ds)

        canon = co.build_doc_kvs(model, [pieces["prefix"] + pieces["df_head"]], device, s)[0]
        assert canon.n_cached == s
        shared_path = root / "shared.safetensors"
        shared_bytes = P.save_shared(canon.to(cpu), shared_path)
        sh_k, sh_v, _ = P.load_shared(shared_path)

        order = sorted(docs, key=lambda d: (n_doc[d], d))
        idx = sorted({round(i * (len(order) - 1) / max(args.sample - 1, 1)) for i in range(args.sample)})
        sample = [order[i] for i in idx]

        per = {f: {"file_bytes": [], "header_bytes": [], "load_ms": [], "checked": 0} for f in FORMATS}
        shared_exact, shared_rel, score_shift = 0, [], []
        for d in sample:
            prefix, _ = co.split_doc_first(pieces, dfart["doc_seam"][d], [], "B")
            kv = co.build_doc_kvs(model, [prefix], device, s)[0]
            kv_cpu = kv.to(cpu)
            own_k = [k[:, :s] for k in kv_cpu.keys]
            own_v = [v[:, :s] for v in kv_cpu.values]

            same = all(torch.equal(a, b) for a, b in zip(own_k + own_v, sh_k + sh_v))
            shared_exact += int(same)
            shared_rel.append(max(rel_diff(a, b) for a, b in zip(own_k + own_v, sh_k + sh_v)))
            mixed = co.DocKV([torch.cat([sk.to(device), k[:, s:]], dim=1) for sk, k in zip(sh_k, kv.keys)],
                             [torch.cat([sv.to(device), v[:, s:]], dim=1) for sv, v in zip(sh_v, kv.values)],
                             kv.next_pos, s)
            qs = by_doc[d][:8]
            news = [co.split_doc_first(pieces, dfart["doc_seam"][d], art["query_df"][q], "B")[1] for q in qs]
            a = co.score_on_caches(model, [kv] * len(qs), news, device)["log_odds"]
            b = co.score_on_caches(model, [mixed] * len(qs), news, device)["log_odds"]
            score_shift.append(max(abs(x - y) for x, y in zip(a, b)))
            del mixed

            for f in FORMATS:
                q_ref = Q.quantize_doc(kv_cpu, f)
                q_mps = Q.quantize_doc(kv, f)
                assert all(torch.equal(x.cpu(), y) for x, y in zip(q_mps.keys + q_mps.values, q_ref.keys + q_ref.values)), \
                    f"MPS and CPU quantize differ ({f}, {d})"
                path = root / f / f"{d.replace('/', '_')}.safetensors"
                info = P.pack_doc(kv_cpu, f, path)
                pred = P.predicted_file_bytes(f, n_doc[d], s, L, H, D, "bfloat16")
                assert info["file_bytes"] == pred, f"file size {info['file_bytes']} != predicted {pred} ({f}, {d})"
                assert info["payload_bytes"] == Q.theory_bytes(f, n_doc[d], H, D, L), f"payload != theory ({f}, {d})"
                with Timer(cpu) as t:
                    u = P.unpack_doc(path, own_k, own_v)
                assert u.next_pos == q_ref.next_pos and all(
                    torch.equal(x, y) for x, y in zip(u.keys + u.values, q_ref.keys + q_ref.values)), \
                    f"unpack != quantize_doc ({f}, {d})"
                per[f]["file_bytes"].append(info["file_bytes"])
                per[f]["header_bytes"].append(info["header_bytes"])
                per[f]["load_ms"].append(t.ms)
                per[f]["checked"] += 1
                if not args.keep:
                    path.unlink()
            del kv, kv_cpu

        files = {f: [P.predicted_file_bytes(f, n_doc[d], s, L, H, D, "bfloat16") for d in docs] for f in FORMATS}
        bf_mean = float(np.mean(files["bf16"]))
        bf_total = sum(files["bf16"]) + shared_bytes
        naive = float(np.mean([Q.theory_bytes("bf16", n_doc[d] + s, H, D, L) for d in docs]))
        fmts = {}
        for f in FORMATS:
            mean_f = float(np.mean(files[f]))
            total = sum(files[f]) + shared_bytes
            fmts[f] = {"file_bytes_per_doc_mean": mean_f, "file_mib_per_doc_mean": mean_f / MiB,
                       "header_bytes_max": max(per[f]["header_bytes"]),
                       "ratio_vs_bf16_file": bf_mean / mean_f,
                       "corpus_total_bytes": total, "corpus_total_gib": total / 1024 ** 3,
                       "ratio_vs_bf16_corpus": bf_total / total,
                       "ratio_vs_naive_bf16": naive / (mean_f + shared_bytes / len(docs)),
                       "sample_checked": per[f]["checked"], "sample_load_ms_warm": summarize_ms(per[f]["load_ms"])}
        r = {"num_docs": len(docs), "shared_len": s, "shared_file_bytes": shared_bytes,
             "doc_tokens": {"min": min(n_doc.values()), "max": max(n_doc.values()),
                            "mean": float(np.mean(list(n_doc.values())))},
             "naive_bf16_bytes_per_doc": naive, "sample_docs": sample,
             "shared_prefix_check": {"bit_identical": [shared_exact, len(sample)], "max_rel_diff": max(shared_rel),
                                     "score_shift_max_abs": max(score_shift),
                                     "score_shift_mean": float(np.mean(score_shift))},
             "formats": fmts}
        report["datasets"][ds] = r
        sc = r["shared_prefix_check"]
        print(f"[{ds}] {len(docs)} docs, doc tokens {r['doc_tokens']['min']}..{r['doc_tokens']['max']} "
              f"(mean {r['doc_tokens']['mean']:.1f}), shared file {shared_bytes / MiB:.2f} MiB, "
              f"naive bf16 {naive / MiB:.2f} MiB/doc")
        print(f"[{ds}] shared prefix: bit-identical {sc['bit_identical']}, max rel {sc['max_rel_diff']:.2e}, "
              f"score shift with canonical prefix max {sc['score_shift_max_abs']:.3f} mean {sc['score_shift_mean']:.3f}")
        for f in FORMATS:
            x = fmts[f]
            print(f"[{ds}] {f:10s} {x['file_mib_per_doc_mean']:7.3f} MiB/doc  header <= {x['header_bytes_max']} B  "
                  f"{x['ratio_vs_bf16_file']:.3f}x file  {x['ratio_vs_bf16_corpus']:.3f}x corpus "
                  f"({x['corpus_total_gib']:.2f} GiB)  {x['ratio_vs_naive_bf16']:.3f}x vs naive  "
                  f"load p50 {x['sample_load_ms_warm']['p50']:.0f} ms  checked {x['sample_checked']}")

    save_json(report, R / "metrics" / "storage.json")
    print("all sampled files: size == predicted, payload == theory, unpack == fake quant (bit-exact)")


if __name__ == "__main__":
    main()