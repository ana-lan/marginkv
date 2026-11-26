# src/marginkv/train/lora.py
"""LoRA training pieces: group -> sequences, grad-enabled yes/no scoring,
listwise loss, dev eval, init check, fp32 merge."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from marginkv.models import qwen3_reranker as qr
from marginkv.train import data as td

ORDERS = ("qf", "df")


def group_seqs(art: dict, qid: str, order: str) -> list[list[int]]:
    """The group's sequences, positive first, in the given prompt order."""
    pieces = art["pieces"]
    if order == "qf":
        return [qr.query_first_ids(pieces, art["query_qf"][qid], art["doc"][p]) for p in art["groups"][qid]]
    if order == "df":
        return [qr.doc_first_seam_ids(pieces, art["doc_seam"][p], art["query_df"][qid]) for p in art["groups"][qid]]
    raise ValueError(order)


def causal_lm(model):
    """The CausalLM (with LoRA layers injected, if any) behind a PeftModel or a plain model."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def forward_log_odds(model, seqs, device) -> torch.Tensor:
    """Same math as qr.score_batch, with gradients: backbone, last position, fp32 yes/no head."""
    lm = causal_lm(model)
    ids, mask, pos = qr.collate(seqs, qr.PAD_ID, device)
    h = lm.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state[:, -1]
    w = lm.lm_head.weight[[qr.NO_ID, qr.YES_ID]]
    logits = h.float() @ w.float().T
    return logits[:, 1] - logits[:, 0]


def listwise_loss(scores: torch.Tensor, group_size: int, temperature: float = 1.0) -> torch.Tensor:
    s = scores.view(-1, group_size) / temperature
    target = torch.zeros(s.shape[0], dtype=torch.long, device=s.device)
    return F.cross_entropy(s, target)


@torch.no_grad()
def evaluate_groups(model, art, qids, order, device, batch_seqs: int = 64) -> dict:
    """Dev metrics within groups: MRR of the positive (ties count against it), acc@1, loss."""
    model.eval()
    G = art["group_size"]
    seqs = [s for q in qids for s in group_seqs(art, q, order)]
    order_idx = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = [0.0] * len(seqs)
    for i in range(0, len(order_idx), batch_seqs):
        b = order_idx[i:i + batch_seqs]
        lo = forward_log_odds(model, [seqs[j] for j in b], device).cpu().tolist()
        for j, v in zip(b, lo):
            out[j] = v
    s = torch.tensor(out).view(-1, G)
    rank = 1 + (s[:, 1:] >= s[:, :1]).sum(1)
    loss = F.cross_entropy(s, torch.zeros(len(s), dtype=torch.long)).item()
    return {"n": len(qids), "mrr": round((1.0 / rank.float()).mean().item(), 4),
            "acc1": round((rank == 1).float().mean().item(), 4), "loss": round(loss, 4)}


@torch.no_grad()
def init_check(model, art, qids, order, device) -> dict:
    """LoRA at init (B = 0) must score exactly like the base model through qr.score_batch."""
    model.eval()
    seqs = [s for q in qids for s in group_seqs(art, q, order)]
    ours = forward_log_odds(model, seqs, device).cpu()
    with model.disable_adapter():
        ref = qr.score_batch(causal_lm(model), seqs, device)["log_odds"]
    ref = torch.as_tensor(ref, dtype=torch.float32).cpu()
    return {"n": len(seqs), "max_abs_diff": float((ours - ref).abs().max())}


def merge_adapter(name, revision, adapter_dir, out_dir, art, check_qids, order) -> dict:
    """Merge on CPU in fp32, check against the unmerged model, then save bf16 safetensors."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    cpu = torch.device("cpu")
    base = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch.float32).eval()
    pm = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
    seqs = [s for q in check_qids for s in group_seqs(art, q, order)]
    with torch.no_grad():
        unmerged = forward_log_odds(pm, seqs, cpu)
        merged = pm.merge_and_unload().eval()
        merged_fp32 = forward_log_odds(merged, seqs, cpu)
    merged = merged.to(torch.bfloat16)
    out_dir = Path(out_dir)
    merged.save_pretrained(out_dir, safe_serialization=True)
    with torch.no_grad():
        merged_bf16 = forward_log_odds(merged, seqs, cpu)
    files = {p.name: td.sha256_file(p) for p in sorted(out_dir.glob("*.safetensors"))}
    return {
        "adapter": str(adapter_dir), "out_dir": str(out_dir), "check_seqs": len(seqs),
        "max_abs_diff_merged_fp32_vs_unmerged": float((merged_fp32 - unmerged).abs().max()),
        "max_abs_diff_bf16_vs_fp32": float((merged_bf16 - merged_fp32).abs().max()),
        "files_sha256": files,
    }