# src/marginkv/models/qwen3_reranker.py
"""Qwen3-Reranker prompt pieces, tokenization, and (from 2.3) yes/no scoring.

Prompt layout follows the model card:
    prefix + "<Instruct>: {inst}\n<Query>: {query}\n<Document>: {doc}" + suffix

We tokenize the prompt in pieces so a document's ids are identical in every prompt
order (needed for KV reuse in Phase 3). Piece boundaries sit where Qwen's pretokenizer
splits anyway, so query-first concatenation equals the model card's tokenization.
scripts/02_tokenize.py verifies this on every reranked pair.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

from transformers import AutoTokenizer, PreTrainedTokenizerBase

SYSTEM_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

YES_ID, NO_ID, PAD_ID = 9693, 2152, 151643
SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")


def load_tokenizer(name: str, revision: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(name, revision=revision)


def encode(tok: PreTrainedTokenizerBase, texts: list[str]) -> list[list[int]]:
    """Batch encode with no special tokens added. Special-token strings in text are still parsed."""
    if not texts:
        return []
    return tok(texts, add_special_tokens=False)["input_ids"]


# Variable pieces carry their leading space (so " word" stays one pretoken).
# The query-first query piece also carries the trailing "\n", because the pretokenizer
# merges punctuation with a following newline (".\n"); splitting there would break parity.
def query_text_qf(query: str) -> str:
    return " " + query + "\n"


def query_text_df(query: str) -> str:
    return " " + query  # followed by <|im_end|>, a special token, so no merge risk


def doc_piece_text(text: str) -> str:
    return " " + text


def piece_texts(instruction: str = DEFAULT_INSTRUCTION) -> dict[str, str]:
    return {
        "prefix": SYSTEM_PREFIX,
        "qf_head": f"<Instruct>: {instruction}\n<Query>:",
        "qf_mid": "<Document>:",
        "df_head": f"<Instruct>: {instruction}\n<Document>:",
        "df_mid": "\n<Query>:",
        "suffix": SUFFIX,
    }


def build_pieces(tok: PreTrainedTokenizerBase, instruction: str = DEFAULT_INSTRUCTION) -> dict[str, list[int]]:
    texts = piece_texts(instruction)
    names = list(texts)
    return dict(zip(names, encode(tok, [texts[n] for n in names])))


def check_tokenizer(tok: PreTrainedTokenizerBase, pieces: dict[str, list[int]]) -> None:
    """Fail loudly if ids we rely on changed or special tokens got split into text."""
    for t, expect in (("yes", YES_ID), ("no", NO_ID)):
        got = tok.convert_tokens_to_ids(t)
        assert got == expect, f"{t!r} id {got} != {expect}"
    assert tok.pad_token_id == PAD_ID, f"pad id {tok.pad_token_id} != {PAD_ID}"
    fixed = pieces["prefix"] + pieces["suffix"]
    for t in SPECIAL_TOKENS:
        tid = tok.convert_tokens_to_ids(t)
        assert tid is not None and tid >= tok.vocab_size, f"{t!r} is not an added special token"
        assert tid in fixed, f"{t!r} (id {tid}) not found as a single id in prefix/suffix"
    assert fixed.count(tok.convert_tokens_to_ids("<|im_start|>")) == 3  # system, user, assistant


def query_first_ids(pieces: dict, q_qf: list[int], doc: list[int]) -> list[int]:
    return pieces["prefix"] + pieces["qf_head"] + q_qf + pieces["qf_mid"] + doc + pieces["suffix"]


def doc_first_ids(pieces: dict, doc: list[int], q_df: list[int]) -> list[int]:
    # Everything up to and including `doc` is the cacheable prefix in Phase 3.
    return pieces["prefix"] + pieces["df_head"] + doc + pieces["df_mid"] + q_df + pieces["suffix"]


def official_query_first_ids(tok, instruction: str, queries: list[str], docs: list[str]) -> list[list[int]]:
    """Model card tokenization: prefix_tokens + tokenizer(middle) + suffix_tokens.
    Uses the tokenizer's default add_special_tokens, as the card does."""
    pre = tok.encode(SYSTEM_PREFIX, add_special_tokens=False)
    suf = tok.encode(SUFFIX, add_special_tokens=False)
    mids = [f"<Instruct>: {instruction}\n<Query>: {q}\n<Document>: {d}" for q, d in zip(queries, docs)]
    return [pre + m + suf for m in tok(mids)["input_ids"]]


def natural_doc_first_ids(tok, instruction: str, queries: list[str], docs: list[str]) -> list[list[int]]:
    """Doc-first string tokenized in one call. Diagnostic only: our doc-first is defined by concatenation."""
    pre = tok.encode(SYSTEM_PREFIX, add_special_tokens=False)
    suf = tok.encode(SUFFIX, add_special_tokens=False)
    mids = [f"<Instruct>: {instruction}\n<Document>: {d}\n<Query>: {q}" for q, d in zip(queries, docs)]
    return [pre + m + suf for m in tok(mids)["input_ids"]]


def load_model(name: str, revision: str, device: torch.device, dtype: torch.dtype):
    """Pinned revision, eval mode. Attention = transformers default (sdpa)."""
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype)
    return model.to(device).eval()


def collate(seqs: list[list[int]], pad_id: int = PAD_ID, device: torch.device | str = "cpu"):
    """Left pad. position_ids = cumsum(mask) - 1, so real tokens start at 0 in every row.
    Pad positions are clamped to 0; they are masked, so the value is irrelevant."""
    if not seqs or any(len(s) == 0 for s in seqs):
        raise ValueError("collate needs non-empty sequences")
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s, dtype=torch.long)
        mask[i, L - len(s):] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids.to(device), mask.to(device), pos.to(device)


@torch.inference_mode()
def score_batch(model, seqs: list[list[int]], device) -> dict[str, list[float]]:
    """One forward pass. yes/no logits are computed in fp32 from the final hidden state:
    bf16 logits at |x| ~ 8 to 16 step in 0.0625 to 0.125, which creates ranking ties.
    The backbone still runs in the model dtype. Identical to the plain forward in fp32."""
    ids, mask, pos = collate(seqs, PAD_ID, device)
    out = model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
    h = out.last_hidden_state[:, -1, :].float()                 # final norm already applied
    w = model.lm_head.weight[[NO_ID, YES_ID]].float()           # [2, hidden]
    logits = h @ w.T                                            # [batch, 2] = (no, yes)
    no, yes = logits[:, 0], logits[:, 1]
    log_odds = yes - no
    p_yes = torch.sigmoid(log_odds)                             # == softmax([no, yes])[1]
    return {
        "logit_yes": yes.cpu().tolist(),
        "logit_no": no.cpu().tolist(),
        "log_odds": log_odds.cpu().tolist(),
        "p_yes": p_yes.cpu().tolist(),
    }


def score_sequences(model, seqs: list[list[int]], batch_size: int, device, on_batch=None) -> dict[str, list[float]]:
    """Score many prompts, batching by length to cut padding. Output is in input order.
    on_batch(done, total) is an optional progress callback."""
    order = sorted(range(len(seqs)), key=lambda i: (len(seqs[i]), i))
    keys = ("logit_yes", "logit_no", "log_odds", "p_yes")
    out = {k: [None] * len(seqs) for k in keys}
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        res = score_batch(model, [seqs[i] for i in idx], device)
        for k in keys:
            for j, i in enumerate(idx):
                out[k][i] = res[k][j]
        if on_batch is not None:
            on_batch(min(start + batch_size, len(order)), len(order))
    return out