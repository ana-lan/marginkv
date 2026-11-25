# src/marginkv/models/bge_reranker.py
"""bge-reranker-v2-m3 cross-encoder baseline.

Input: [cls] query [sep][sep] doc [sep], doc truncated to max_doc_tokens of bge's own tokenizer.
Score: classifier logit on the first token. Backbone runs in the model dtype; the small
classifier head runs in fp32 so the score isn't quantized to bf16 steps (same as Qwen3's yes/no head).
"""
from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase


def load_tokenizer(name: str, revision: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(name, revision=revision)


def load_model(name: str, revision: str, device: torch.device, dtype: torch.dtype):
    model = AutoModelForSequenceClassification.from_pretrained(name, revision=revision, dtype=dtype)
    assert model.config.num_labels == 1, f"expected 1 label, got {model.config.num_labels}"
    assert hasattr(model, "classifier"), "expected an XLM-R style .classifier head"
    model = model.to(device).eval()
    model.classifier.float()  # fp32 head; only our score_batch path is used with this model
    return model


def encode(tok: PreTrainedTokenizerBase, texts: list[str]) -> list[list[int]]:
    if not texts:
        return []
    return tok(texts, add_special_tokens=False)["input_ids"]


def pair_ids(tok: PreTrainedTokenizerBase, q_ids: list[int], d_ids: list[int]) -> list[int]:
    """XLM-R pair layout: <s> A </s></s> B </s>."""
    cls, sep = tok.cls_token_id, tok.sep_token_id
    return [cls] + q_ids + [sep, sep] + d_ids + [sep]


def collate(seqs: list[list[int]], pad_id: int, device):
    """Right pad (encoder norm; RoBERTa derives positions from non-pad tokens)."""
    if not seqs or any(len(s) == 0 for s in seqs):
        raise ValueError("collate needs non-empty sequences")
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)


@torch.inference_mode()
def score_batch(model, seqs: list[list[int]], pad_id: int, device) -> dict[str, list[float]]:
    ids, mask = collate(seqs, pad_id, device)
    h = model.base_model(input_ids=ids, attention_mask=mask).last_hidden_state
    logit = model.classifier(h[:, :1, :].float())[:, 0]  # head reads position 0; fp32
    return {"logit": logit.cpu().tolist(), "p": torch.sigmoid(logit).cpu().tolist()}


def score_sequences(model, seqs: list[list[int]], pad_id: int, batch_size: int, device, on_batch=None):
    """Length-sorted batching; output in input order."""
    order = sorted(range(len(seqs)), key=lambda i: (len(seqs[i]), i))
    out = {"logit": [None] * len(seqs), "p": [None] * len(seqs)}
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        res = score_batch(model, [seqs[i] for i in idx], pad_id, device)
        for k in out:
            for j, i in enumerate(idx):
                out[k][i] = res[k][j]
        if on_batch is not None:
            on_batch(min(start + batch_size, len(order)), len(order))
    return out