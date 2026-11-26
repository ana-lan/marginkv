# scripts/06_train_lora.py
"""Phase 4.2/4.3: LoRA on Qwen3-Reranker-0.6B, query-first control (qf) or doc-first variant B (df).

Listwise CE over groups (1 pos + 7 neg) on the fp32 yes/no log-odds. Both arms use
the same groups, order, steps and seed; only the prompt order differs.
Checkpoint = best dev MRR among trained steps (pre-registered). Merge in fp32, save bf16.

  --check-init    only verify LoRA at init == base scorer
  --overfit N     train and eval on the same N groups (sanity), under artifacts/smoke
  --smoke N       N train groups, small dev, under artifacts/smoke (includes merge)
  --stop-at K     exit after step K (resume test)
  --merge-only    merge artifacts/.../best into .../merged
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import math
import random
import sys
import time
from dataclasses import asdict

import peft
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

from marginkv.config import load_config
from marginkv.models import qwen3_reranker as qr
from marginkv.train import data as td
from marginkv.train import lora as tl
from marginkv.utils.device import get_device, get_dtype, sync
from marginkv.utils.io import ensure_dir, load_json, load_pt, run_info, save_json, save_pt
from marginkv.utils.seed import set_seed


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", choices=tl.ORDERS, required=True)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--check-init", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--micro-groups", type=int, default=0)
    ap.add_argument("--dev-every", type=int, default=0)
    ap.add_argument("--dev-limit", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--stop-at", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore an existing checkpoint")
    return ap.parse_args()


def lr_lambda(total, warmup):
    def f(s):
        if s < warmup:
            return (s + 1) / warmup
        return max(0.0, (total - s) / max(1, total - warmup))
    return f


class StepGroups:
    """Deterministic data order: shuffle per epoch with seed + epoch, slice per step."""

    def __init__(self, qids, gps, seed):
        self.qids, self.gps, self.seed = list(qids), gps, seed
        self.spe = len(qids) // gps
        self._epoch, self._perm = None, None

    def __call__(self, step):
        epoch, within = divmod(step, self.spe)
        if epoch != self._epoch:
            self._perm = list(self.qids)
            random.Random(self.seed + epoch).shuffle(self._perm)
            self._epoch = epoch
        return self._perm[within * self.gps:(within + 1) * self.gps]


def do_merge(cfg, art, dev_q, order, out, res_path):
    src = out / "best"
    if not (src / "adapter_config.json").exists():
        sys.exit(f"no adapter at {src}")
    t = time.time()
    print(f"merging {src} on CPU fp32 ...", flush=True)
    rep = tl.merge_adapter(cfg.models.qwen3.name, cfg.models.qwen3.revision, src, out / "merged",
                           art, dev_q[:4], order)
    rep["seconds"] = round(time.time() - t, 1)
    save_json({"base": {"name": cfg.models.qwen3.name, "revision": cfg.models.qwen3.revision},
               "order": order, **rep, "run_info": run_info()}, out / "merged" / "marginkv_merge.json")
    res = load_json(res_path) if res_path.exists() else {}
    res["merge"] = rep
    save_json(res, res_path)
    print(f"merge: fp32 merged vs unmerged {rep['max_abs_diff_merged_fp32_vs_unmerged']:.3g}, "
          f"bf16 vs fp32 {rep['max_abs_diff_bf16_vs_fp32']:.3g} ({rep['check_seqs']} seqs) -> {out / 'merged'}")


def main():
    args = parse()
    cfg = load_config()
    L = cfg.lora
    set_seed(cfg.seed)
    device = get_device(cfg.runtime.device)
    dtype = get_dtype(cfg.runtime.dtype)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    small = bool(args.smoke or args.overfit)
    base = cfg.paths.artifacts_dir / "smoke" if small else cfg.paths.artifacts_dir
    out = ensure_dir(base / "lora" / args.order)
    res_path = (base / "lora" / f"train_{args.order}.json") if small \
        else (cfg.paths.results_dir / "lora" / f"train_{args.order}.json")

    art = load_pt(cfg.paths.artifacts_dir / "train" / "msmarco_qwen3.pt")
    G = art["group_size"]
    assert G == L.group_size, (G, L.group_size)
    train_q, dev_q = art["split"]["train"], art["split"]["dev"]
    if args.overfit:
        train_q = dev_q = train_q[: args.overfit]
    elif args.smoke:
        train_q, dev_q = train_q[: args.smoke], dev_q[: max(8, args.smoke // 2)]
    if args.dev_limit:
        dev_q = dev_q[: args.dev_limit]

    if args.merge_only:
        do_merge(cfg, art, dev_q, args.order, out, res_path)
        return

    gps = min(L.groups_per_step, len(train_q))
    micro = args.micro_groups or L.micro_groups
    assert gps % micro == 0, (gps, micro)
    lr = args.lr or L.lr
    dev_every = args.dev_every or L.dev_every
    total = args.max_steps or L.epochs * (len(train_q) // gps)
    warmup = max(1, math.ceil(L.warmup_frac * total))

    # model
    lm = qr.load_model(cfg.models.qwen3.name, cfg.models.qwen3.revision, device, dtype)
    lm.config.use_cache = False
    if args.grad_ckpt:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pcfg = LoraConfig(r=L.r, lora_alpha=L.alpha, lora_dropout=L.dropout,
                      target_modules=list(L.target_modules), bias="none")
    model = get_peft_model(lm, pcfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"order {args.order}  trainable {n_tr:,} / {n_all:,} ({100 * n_tr / n_all:.2f}%)  "
          f"adapter dtype {trainable[0].dtype}  device {device}", flush=True)

    init = tl.init_check(model, art, dev_q[:2], args.order, device)
    print(f"init check: max |d log-odds| {init['max_abs_diff']:.3g} over {init['n']} seqs", flush=True)
    if init["max_abs_diff"] > 1e-3:
        sys.exit("LoRA at init does not match the base scorer")
    if args.check_init:
        return

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=L.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total, warmup))
    settings = {"order": args.order, "lora": asdict(L), "lr": lr, "groups_per_step": gps,
                "total_steps": total, "warmup_steps": warmup, "seed": cfg.seed,
                "num_train": len(train_q), "num_dev": len(dev_q),
                "train_sha": td.content_sha256(train_q), "dev_sha": td.content_sha256(dev_q),
                "artifact_sha": art["content_sha256"]}
    settings_sha = td.content_sha256(settings)

    state = {"step": 0, "best": None, "history": [], "dev": [], "resumed_at": []}
    ckpt_path = out / "ckpt.pt"
    if ckpt_path.exists() and not args.force:
        ck = load_pt(ckpt_path)
        if ck["settings_sha"] != settings_sha:
            sys.exit(f"{ckpt_path} has different settings; use --force to restart")
        set_peft_model_state_dict(model, {k: v.to(device) for k, v in ck["adapter"].items()})
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        state = ck["state"]
        state["resumed_at"].append(state["step"])
        print(f"resumed at step {state['step']}", flush=True)

    timing = {"train_s": 0.0, "seqs": 0, "tokens": 0}

    def write_results(extra=None):
        res = {"run_info": run_info(), "peft_version": peft.__version__, "settings": settings,
               "settings_sha": settings_sha, "trainable_params": n_tr, "total_params": n_all,
               "adapter_dtype": str(trainable[0].dtype), "device": str(device),
               "dtype": cfg.runtime.dtype, "micro_groups": micro, "grad_ckpt": args.grad_ckpt,
               "init_check": init, "timing": timing, "state": state, **(extra or {})}
        if device.type == "cuda":
            res["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        save_json(res, ensure_dir(res_path.parent) / res_path.name)

    def run_dev(step):
        t = time.time()
        m = tl.evaluate_groups(model, art, dev_q, args.order, device)
        m.update(step=step, seconds=round(time.time() - t, 1))
        state["dev"].append(m)
        tag = ""
        if step > 0 and (state["best"] is None or m["mrr"] >= state["best"]["mrr"]):
            model.save_pretrained(out / "best")
            state["best"] = m
            tag = "  (best, saved)"
        print(f"  dev @ {step}: mrr {m['mrr']:.4f}  acc1 {m['acc1']:.4f}  loss {m['loss']:.4f}"
              f"  [{m['seconds']} s]{tag}", flush=True)
        model.train()

    if state["step"] == 0 and not state["dev"]:
        run_dev(0)

    groups_at = StepGroups(train_q, gps, cfg.seed)
    model.train()
    start = state["step"]
    t0 = time.time()
    window = []
    for step in range(start, total):
        groups = groups_at(step)
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for i in range(0, len(groups), micro):
            chunk = groups[i:i + micro]
            seqs = [s for q in chunk for s in tl.group_seqs(art, q, args.order)]
            scores = tl.forward_log_odds(model, seqs, device)
            loss = tl.listwise_loss(scores, G, L.temperature) * (len(chunk) / len(groups))
            loss.backward()
            loss_sum += loss.item()
            timing["seqs"] += len(seqs)
            timing["tokens"] += sum(len(s) for s in seqs)
        gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, L.max_grad_norm))
        opt.step()
        sched.step()
        state["step"] = step + 1
        window.append(loss_sum)

        done = step + 1
        if done % args.log_every == 0 or done == total:
            sync(device)
            el = time.time() - t0
            timing["train_s"] = round(el, 1)
            sps = timing["seqs"] / el
            eta = (total - done) * el / (done - start) / 60
            entry = {"step": done, "loss": round(sum(window) / len(window), 4),
                     "lr": sched.get_last_lr()[0], "grad_norm": round(gnorm, 3),
                     "seqs_per_s": round(sps, 1)}
            state["history"].append(entry)
            window = []
            print(f"step {done}/{total}  loss {entry['loss']:.4f}  lr {entry['lr']:.2e}  "
                  f"gnorm {gnorm:.2f}  {sps:.1f} seq/s  eta {eta:.1f} min", flush=True)
        if done % dev_every == 0 or done == total:
            run_dev(done)
        if done % args.ckpt_every == 0 or done == total:
            save_pt({"settings_sha": settings_sha,
                     "adapter": {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()},
                     "opt": opt.state_dict(), "sched": sched.state_dict(), "state": state}, ckpt_path)
            write_results()
        if args.stop_at and done == args.stop_at:
            print(f"stop-at {done}: exiting (rerun without --stop-at to resume)")
            return

    model.save_pretrained(out / "final")
    write_results()
    print(f"done: best dev mrr {state['best']['mrr']:.4f} at step {state['best']['step']}; "
          f"{timing['seqs'] / max(timing['train_s'], 1e-9):.1f} seq/s", flush=True)

    if not args.overfit:
        del model, opt, lm
        if device.type == "cuda":
            torch.cuda.empty_cache()
        do_merge(cfg, art, dev_q, args.order, out, res_path)


if __name__ == "__main__":
    main()