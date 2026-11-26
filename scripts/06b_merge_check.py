# scripts/06b_merge_check.py
"""Phase 4.4 pre-check: does merging the LoRA into bf16 weights add error?

For each arm (qf, df), on 64 fixed SciFact pairs in that arm's prompt order, against the fp32
reference (base fp32 + adapter fp32, unmerged, CPU = exact W + dW):
  merged error    = merged bf16 (MPS)
  unmerged error  = base bf16 + fp32 adapter, unmerged (MPS)
  base error      = base bf16 (MPS) vs base fp32 (CPU)
Rules:
  pass_vs_base (original, pre-registered in 4.2): mean merged err <= 1.5 x mean base err
  pass_merge_only (amended 2026-11-25 before its data was seen): mean merged err <= 1.5 x mean unmerged err
06c uses pass_merge_only. If it fails, evaluate both arms with the unmerged adapter.
Output: results/metrics/lora_merge_check.json
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gc

import torch
from peft import PeftModel

from marginkv.config import load_config
from marginkv.models import qwen3_reranker as qr
from marginkv.train.lora import forward_log_odds
from marginkv.utils.device import get_device, get_dtype
from marginkv.utils.io import load_pt, run_info, save_json
from marginkv.utils.seed import set_seed

N_Q, N_D, BS, RATIO = 8, 8, 8, 1.5


@torch.no_grad()
def score(model, seqs, device):
    model.eval()
    order = sorted(range(len(seqs)), key=lambda i: (len(seqs[i]), i))
    out = torch.empty(len(seqs))
    for s in range(0, len(order), BS):
        b = order[s:s + BS]
        out[b] = forward_log_odds(model, [seqs[i] for i in b], device).float().cpu()
    return out


def free(device):
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


def stats(e):
    return {"mean": round(float(e.mean()), 5), "p95": round(float(e.quantile(0.95)), 5), "max": round(float(e.max()), 5)}


def main():
    cfg = load_config()
    set_seed(cfg.seed)
    dev, bf16, cpu = get_device(cfg.runtime.device), get_dtype(cfg.runtime.dtype), torch.device("cpu")
    name, rev = cfg.models.qwen3.name, cfg.models.qwen3.revision
    ds = cfg.datasets[0]
    art = load_pt(cfg.paths.artifacts_dir / "tokenized" / f"{ds}_qwen3.pt")
    dfart = load_pt(cfg.paths.artifacts_dir / "tokenized" / f"{ds}_qwen3_df.pt")
    pb = {**art["pieces"], **dfart["seam_pieces"]}
    qids = art["query_order"][:N_Q]
    pairs = [(q, d) for q in qids for d in art["rerank_candidates"][q][:N_D]]
    seqs = {
        "qf": [qr.query_first_ids(art["pieces"], art["query_qf"][q], art["doc"][d]) for q, d in pairs],
        "df": [qr.doc_first_seam_ids(pb, dfart["doc_seam"][d], art["query_df"][q]) for q, d in pairs],
    }
    lora = cfg.paths.artifacts_dir / "lora"

    print(f"base bf16 on {dev} ...", flush=True)
    m = qr.load_model(name, rev, dev, bf16)
    base_bf16 = {a: score(m, seqs[a], dev) for a in seqs}
    del m
    free(dev)

    arms = {}
    for a in ("qf", "df"):
        print(f"[{a}] merged bf16 on {dev} ...", flush=True)
        m = qr.load_model(str(lora / a / "merged"), None, dev, bf16)
        merged_bf16 = score(m, seqs[a], dev)
        del m
        free(dev)

        print(f"[{a}] unmerged adapter on bf16 base, {dev} ...", flush=True)
        m = qr.load_model(name, rev, dev, bf16)
        pm = PeftModel.from_pretrained(m, str(lora / a / "best"))
        unmerged_bf16 = score(pm, seqs[a], dev)
        del pm, m
        free(dev)

        print(f"[{a}] base fp32 and base+adapter fp32 on cpu ...", flush=True)
        base32 = qr.load_model(name, rev, cpu, torch.float32)
        base_fp32 = score(base32, seqs[a], cpu)
        pm = PeftModel.from_pretrained(base32, str(lora / a / "best"))
        lora_fp32 = score(pm, seqs[a], cpu)
        del pm, base32
        gc.collect()

        e_base = (base_bf16[a] - base_fp32).abs()
        e_merged = (merged_bf16 - lora_fp32).abs()
        e_unmerged = (unmerged_bf16 - lora_fp32).abs()
        r_base = float(e_merged.mean() / e_base.mean())
        r_unm = float(e_merged.mean() / e_unmerged.mean())
        arms[a] = {
            "base_bf16_err": stats(e_base),
            "unmerged_bf16_err": stats(e_unmerged),
            "merged_bf16_err": stats(e_merged),
            "merged_vs_unmerged_bf16": stats((merged_bf16 - unmerged_bf16).abs()),
            "ratio_vs_base": round(r_base, 3),
            "ratio_vs_unmerged": round(r_unm, 3),
            "lora_shift_fp32": stats((lora_fp32 - base_fp32).abs()),
            "pass_vs_base": r_base <= RATIO,
            "pass_merge_only": r_unm <= RATIO,
        }
        r = arms[a]
        print(f"[{a}] err mean: base {r['base_bf16_err']['mean']:.4f}  unmerged {r['unmerged_bf16_err']['mean']:.4f}  "
              f"merged {r['merged_bf16_err']['mean']:.4f} | ratio vs base {r['ratio_vs_base']:.2f} "
              f"({'PASS' if r['pass_vs_base'] else 'FAIL'})  vs unmerged {r['ratio_vs_unmerged']:.2f} "
              f"({'PASS' if r['pass_merge_only'] else 'FAIL'})  shift {r['lora_shift_fp32']['mean']:.2f}", flush=True)

    out = cfg.paths.results_dir / "metrics" / "lora_merge_check.json"
    save_json({"run_info": run_info(), "dataset": ds, "num_pairs": len(pairs), "batch_size": BS,
               "rule_original": f"mean |merged bf16 - fp32 ref| <= {RATIO} x mean |base bf16 - base fp32| (failed for df)",
               "rule_used": f"mean |merged bf16 - fp32 ref| <= {RATIO} x mean |unmerged bf16 - fp32 ref| "
                            "(amended 2026-11-25 before its data was seen: the original baseline depends on score scale)",
               "device": str(dev), "arms": arms}, out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()