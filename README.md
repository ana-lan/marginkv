# MarginKV: Storage-Efficient KV Cache Reuse for Decoder-Only Reranking

**Results explorer:** [marginkv.vercel.app](https://marginkv.vercel.app), an interactive view of the quantization and serving results (every number is read from `results/` by `scripts/14_export_app_data.py`).

A decoder-only reranker can read each document once and keep its KV cache "in the margins," then reuse it for every query that retrieves that document. This repo measures three things: what document-first scoring costs in ranking quality, how far the stored document KV can be compressed (fewer bits, fewer tokens) before ranking quality drops, and whether the reuse actually pays off when the reranker is served on real traffic.

**Status:** Phases 1 to 6 and the vLLM serving study (8a) are done: document-first adaptation, cache correctness, KV quantization with packed storage, query-independent token pruning, and prefix-cache serving on the real query stream. Combined compression and disk benchmarks are next (see [Roadmap](#roadmap)).

## Results at a glance

- **Qwen3-Reranker-0.6B fails document-first zero-shot.** On SciFact, nDCG@10 drops from 0.756 to 0.286. A LoRA trained on MS MARCO in the doc-first order brings it back to 0.750 (SciFact) and 0.354 (NFCorpus): within 0.011 of a query-first LoRA control trained the same way and 0.005 of the original model on SciFact, and above both on NFCorpus.
- **The cached path is exact.** Scoring on a stored document KV matches full-prompt scoring to 2.4e-5 log-odds in fp32.
- **INT4 (KIVI-style) cuts stored document KV 3.2x with no meaningful quality change:** 8.33 MiB per doc instead of 26.4 MiB, with nDCG@10 +0.0010 on SciFact and −0.0009 on NFCorpus. INT8 (1.97x) shows no measurable loss on either dataset. All sizes are measured from packed files on disk that round-trip bit-exactly.
- **The quantization axis matters more than the bit width.** The same INT4 budget with per-token keys loses 0.04 to 0.10 nDCG@10. Qwen3's post-RoPE keys have outlier channels, so keys need per-channel scales.
- **Dropping tokens is a much weaker lever than dropping bits.** Only 75% token retention (1.33x) stays within one point. At the same ~13 MiB per doc, FP8 and INT8 lose at most 0.0015 nDCG@10 while 50% pruning loses 0.008 to 0.08, and INT4 beats every pruning setting. Importance scores from the document's own attention do no better than random, because they favor the end of the document; keeping its start matters more (a post-hoc finding).
- **In real traffic, cache capacity decides whether reuse pays off.** Served with vLLM on the BEIR query stream, 46 to 62% of documents repeat, but LRU eviction and 64-token blocks leave doc-first serving only 0.16 / 0.29 of prompt tokens from cache. FP8 KV doubles capacity and lifts that to 0.23 / 0.37 at −0.003 / −0.001 nDCG@10. A simple block-level cache simulator predicts every measured hit rate.

## Question

- **Q1.** Can Qwen3-Reranker-0.6B score with the document *before* the query, so the document's KV is query-independent and reusable, without losing ranking quality?
- **Q2.** Given that model, how much can the stored document KV be compressed (quantization, query-independent token pruning) before ranking quality drops?

HyperRAG ([An et al., 2025](https://arxiv.org/abs/2504.02921)) established document-side KV reuse for decoder-only rerankers and discussed low-bit KV quantization, but did not evaluate quantized KV. Most KV compression work targets generation, where the cache lives for one request. Here the cache is query-independent and reused across many queries, so the question is storage: bytes per document vs. ranking quality. AdaptCache ([2025](https://arxiv.org/abs/2509.00105)) is the closest storage-side work, for generation serving.

## Setup

Every experiment uses the same fixed inputs:

- **Data:** BEIR SciFact test (300 queries, 5,183 docs) and NFCorpus test (323 queries, 3,633 docs), from pinned Hugging Face mirrors with per-file sha256. Counts match BEIR's published stats.
- **First stage:** BM25 top 100 (bm25s, k1 1.2, b 0.75), saved once. Recall@100: 0.913 / 0.247.
- **Reranking:** top 20 reranked, then BM25 ranks 21 to 100. Metrics are nDCG@10 and MRR@10.
- **Tokenization:** done once and saved. Docs are truncated to 256 tokens. The query-first prompt matches the model card's tokenization on every pair.
- **Document-first prompt:** `[system + instruction][document] | [query][yes/no suffix]`. Everything before `|` is cached. The score is log-odds of "yes" vs "no" at the last position, computed in fp32.
- **Hardware:** quality experiments run on an M3 MacBook Air (MPS, bf16). LoRA training ($2.13) and the vLLM serving study ($1.76) ran on one rented RTX 4090.

## Results

### 1. Document-first ordering (Q1)

nDCG@10 / MRR@10, full prompt, identical tokens for every system.

| System | SciFact | NFCorpus |
|---|---|---|
| BM25 | 0.6791 / 0.6431 | 0.3187 / 0.5245 |
| Random top-20 order (100 reps) | 0.1996 / 0.1360 | 0.1769 / 0.2797 |
| bge-reranker-v2-m3 | 0.7281 / 0.7026 | 0.3333 / 0.5654 |
| Qwen3-Reranker-0.6B, query-first (original) | 0.7555 / 0.7345 | 0.3485 / 0.5662 |
| Query-first LoRA (control) | 0.7617 / 0.7396 | 0.3504 / 0.5721 |
| Document-first, zero-shot | 0.2855 / 0.2219 | 0.2350 / 0.3705 |
| **Document-first LoRA** | **0.7504 / 0.7239** | **0.3543 / 0.5879** |

LoRA setup: r 16 on all 7 projections, listwise softmax loss over 1 positive + 7 negatives, 12k MS MARCO queries, 1,500 steps. Both arms are trained identically, and the checkpoint is chosen on held-out MS MARCO dev only.

**Order gate**, pre-registered as doc-first LoRA ≥ control − 0.01 on both datasets:
- **NFCorpus passes:** +0.0039, 95% CI [−0.0016, +0.0096].
- **SciFact misses by 0.0013:** −0.0113, CI [−0.0240, +0.0011]. The CI includes zero, and the doc-first LoRA is within 0.005 of the original model there.

This is reported as a miss. A length-matched retrain is planned as an ablation, reported next to this result rather than replacing it.

### 2. Cache correctness

- **fp32:** cached scoring vs full-prompt scoring agree to 2.4e-5 log-odds, with identical rankings.
- **bf16:** the two paths differ by rounding (up to about 0.3 log-odds) but have the same error against fp32 (0.0533 vs 0.0532): an exact tie that technically fails our pre-registered point-estimate rule by 1e-4, reported as-is. Their nDCG@10 differs by +0.0002 / −0.0006.

All compression results below compare arms within the cached path, so this offset cancels.

### 3. KV quantization (Q2)

Each doc's KV is built once, quantized, dequantized back to bf16 ("fake quant"), and scored. Every format sees identical pairs in identical batches. Only document tokens are quantized; the shared 59-token system and instruction prefix stays BF16 and is stored once per corpus.

**Noise floor and margin.** Before any compressed run, we scored the BF16 baseline at two batch sizes. That moved scores by up to 0.29 log-odds but nDCG@10 by at most 0.0002. The pre-registered equivalence margin is ±0.002 nDCG@10.
- **No measurable loss:** the format's paired 95% CI sits inside the margin.
- **Within 1 point:** the point estimate is ≥ −0.01.

| Format | SciFact nDCG@10 (Δ, 95% CI) | NFCorpus nDCG@10 (Δ, 95% CI) | Verdict (Sci / NFC) | MiB per doc | vs BF16 |
|---|---|---|---|---|---|
| BF16 | 0.7506 | 0.3538 | baseline | 26.40 | 1.00x |
| FP8 e4m3, per-tensor scale | 0.7494 (−0.0013 [−0.0042, +0.0006]) | 0.3532 (−0.0006 [−0.0017, +0.0006]) | 1 pt / none | 13.20 | 2.00x |
| FP8 e4m3, per-head scale | 0.7491 (−0.0015 [−0.0043, −0.0001]) | 0.3541 (+0.0004 [−0.0010, +0.0020]) | 1 pt / none | 13.20 | 2.00x |
| INT8, per-token | 0.7505 (−0.0002 [−0.0008, +0.0004]) | 0.3537 (−0.0001 [−0.0015, +0.0013]) | none / none | 13.41 | 1.97x |
| **INT4 KIVI-style, g32** | **0.7517 (+0.0010 [−0.0007, +0.0038])** | **0.3528 (−0.0009 [−0.0027, +0.0007])** | 1 pt / 1 pt | **8.33** | **3.17x** |
| INT4 per-token, g32 (diagnostic) | 0.6540 (−0.0966 [−0.120, −0.074]) | 0.3103 (−0.0434 [−0.055, −0.032]) | loss / loss | 8.25 | 3.20x |

"none" = no measurable loss; "1 pt" = within 1 point. KIVI-style INT4 means keys quantized per channel (groups of 32 tokens) and values per token (groups of 32 channels), with fp16 scales and mins.

**Storage** comes from real packed files, not arithmetic:
- **Format:** one safetensors file per doc, INT4 two codes per byte, fp16 scales, plus one shared-prefix file per corpus (6.45 MiB). Headers are under 700 bytes.
- **Round trip:** every sampled file unpacks bit-exactly to the tensors that were scored, and file sizes match the predicted byte counts exactly.
- **Corpus totals:** BF16 needs 82.9 GiB for SciFact's 3,216 reranked docs; INT4 KIVI needs 26.2 GiB.
- **Naive baseline:** caching the full prompt prefix per doc in BF16 takes 32.9 MiB per doc, so INT4 KIVI is 3.9x smaller than that.

#### What we learned

1. **The quantization axis beats bit width.** INT4 keys quantized per channel are as accurate as INT8 keys quantized per token (relative error 0.0230 vs 0.0225). Per-token INT4 keys shift scores by up to 20 log-odds.
2. **Relative KV error is a poor proxy for ranking quality.** Per-token INT4 keys are "only" about 4.6x worse in relative error but lose up to 0.10 nDCG@10, while FP8's uniform 2.7% error barely moves rankings. Measure the task metric.
3. **INT8 is slightly safer than FP8 at the same size.** FP8 per-head shows a small but significant SciFact loss (−0.0015, CI excludes 0); INT8 shows none. Per-tensor and per-head FP8 tie, as expected for a float format.
4. **Storing the shared prefix once is free compression.** 59 of about 300 tokens are identical across docs, which alone saves 1.24x.

### 4. Token pruning (Q2)

Instead of fewer bits per value, keep fewer tokens. After a doc's KV is built, a fraction of its document tokens is dropped from the cache. Kept tokens keep their original positions (RoPE is already applied to the stored keys), so the query starts exactly where it always did. The shared prefix is never pruned, and the first doc token and the last 8 are always kept. In fp32, scoring on a pruned cache matches a full prompt with the pruned tokens hidden from the query to 2.7e-5 log-odds, so every difference below comes from which tokens are dropped.

Methods, each at 25 / 50 / 75% of document tokens:
- **Uniform:** evenly spaced tokens.
- **Boundary:** the first 4 tokens plus the most recent ones (StreamingLLM-style).
- **Attention (a):** the mean attention a token receives from later document tokens, averaged over all heads and layers. It is computed once per document with no query in sight, so it is query-independent.
- **Attention (b):** (a) weighted by position, meant to cancel the bias that uniform causal attention would create.
- **Random:** seeded per document.
- **H2O-style (diagnostic):** plain accumulated attention (Σ of attention received, no normalization). Reported, never selected.

Same harness, baseline, margin and verdict rules as quantization. The rules, including how a setting is picked for Phase 7, were committed before the full run. Δ nDCG@10 vs BF16 (bold = within 1 point):

| Method | SciFact 75% | 50% | 25% | NFCorpus 75% | 50% | 25% |
|---|---|---|---|---|---|---|
| Uniform | **−0.0014** | −0.0122 | −0.0617 | **−0.0006** | **−0.0080** | −0.0194 |
| Random | **−0.0009** | −0.0146 | −0.0818 | **+0.0002** | **−0.0079** | −0.0287 |
| Attention (a) | **−0.0092** | −0.0294 | −0.0875 | **−0.0063** | −0.0138 | −0.0385 |
| Attention (b) | −0.0101 | −0.0416 | −0.1637 | **−0.0091** | −0.0231 | −0.0604 |
| Boundary | −0.0310 | −0.0822 | −0.1488 | −0.0161 | −0.0314 | −0.0504 |
| H2O-style (diagnostic) | **−0.0049** | −0.0125 | −0.0383 | **−0.0014** | **−0.0028** | −0.0113 |
| **MiB per doc (vs BF16)** | 19.8 (1.33x) | 13.2 (1.99x) | 6.6 (4.01x) | 19.8 (1.33x) | 13.2 (1.99x) | 6.6 (4.01x) |

**Bits vs tokens at equal storage** (packed BF16 files for the pruned caches, exact sizes):

| Storage per doc | Quantization | Δ SciFact / NFCorpus | Pruning (uniform) | Δ SciFact / NFCorpus |
|---|---|---|---|---|
| ~13.2 MiB | FP8 per-tensor | −0.0013 / −0.0006 | 50% | −0.0122 / −0.0080 |
| ~13.3 MiB | INT8 per-token | −0.0002 / −0.0001 | 50% | −0.0122 / −0.0080 |
| 8.3 vs 6.6 MiB | INT4 KIVI | +0.0010 / −0.0009 | 25% | −0.0617 / −0.0194 |

#### What we learned

1. **Quantization beats pruning at every storage size we measured.** INT4 KIVI at 8.33 MiB per doc loses less than 75% pruning does at 19.8 MiB. The only pruning setting within one point on both datasets is 75% retention (uniform or random), which saves just 1.33x.
2. **Attention-based importance doesn't help here.** (a) ties random at every retention level. (b) is significantly worse than random at 25% and 50% on both datasets, and the pre-registered (b)-vs-(a) comparison comes out "hurts" at 25%. The reason is in the scores themselves: document tokens attend mostly to nearby earlier tokens, and normalizing by the number of later tokens turns that into a preference for the end of the document (rank correlation with position +0.52 for (a), +0.73 for (b)). Boundary, which keeps the most recent tokens, is the worst method at every retention.
3. **Keeping the start of the document matters (post hoc).** The share of the first 32 doc tokens a method keeps tracks its quality at every retention. At 25%, H2O-style keeps 69% / 66% of them and loses least (−0.038 / −0.011); attention (b) keeps 4% and loses most (−0.164 / −0.060). H2O-style is the only score that beats random (at 25%, both datasets: +0.044 [+0.020, +0.067] and +0.017 [+0.007, +0.028]). A plausible reason is that each document starts with its title. This was not pre-registered; a "keep the first N tokens" arm (equivalent to truncating documents shorter) is the clean follow-up.
4. **Scores move a lot even when rankings don't.** At 75% retention, scores shift by up to 6 to 14 log-odds and 4 to 7.5% of pairs swap order (vs 0.3 log-odds and 0.4% for bf16 noise), yet uniform pruning stays within 0.0015 nDCG@10.
5. **Pruning isn't a speed lever here.** Scoring at 25% retention is only 15 to 23% faster per batch on the Mac, because fixed per-batch costs dominate at this size.

About half of every document token's attention goes to the first token of the shared prefix (an attention sink), which is never pruned. Importance rankings are identical whether computed in MPS bf16 or CPU fp32. **Phase 7 setting (pre-registered rule):** uniform 75%, combined with FP8 and INT4 KIVI.

### 5. Serving with vLLM

Does document-first KV reuse actually pay off when the reranker is served for real? We served the models with vLLM 0.34.0 on one RTX 4090 (24 GB, RunPod), with automatic prefix caching on and vLLM's defaults otherwise (including the block size it picked, 64 tokens). Scores come from the yes/no logprobs of a single generated token (`allowed_token_ids = [no, yes]`, processed logprobs, the model's own sampling config ignored), using the same token ids as every earlier phase.

Settings: **query-first** (original Qwen3-Reranker-0.6B), **doc-first** (merged doc-first LoRA), and **doc-first with FP8 KV** (`--kv-cache-dtype fp8`, vLLM's default scales). The gate rules were committed before any vLLM run.

**The served model is the same model.** Against CPU fp32 on 256 pairs, vLLM's mean |Δ log-odds| is 0.051 (doc-first) and 0.052 (query-first), vs 0.053 and 0.085 for our own MPS bf16 path. With BF16 KV, full-test-set nDCG@10 is within ±0.0007 of the HF scores at every concurrency.

**Trace 3, the real BEIR query stream:** queries in file order, each with its 20 rerank candidates (6,000 / 6,460 requests), cold cache, concurrency 16.

| | SciFact | NFCorpus |
|---|---|---|
| Documents that repeat in the stream | 46.4% | 61.7% |
| Median gap before a document returns | 975 requests | 440 requests |
| **Prompt tokens served from cache** | | |
| Query-first, BF16 KV (177,664-token cache) | 0.183 | 0.198 |
| Doc-first, BF16 KV (177,664-token cache) | 0.159 | 0.286 |
| Doc-first, FP8 KV (340,992-token cache) | 0.225 | 0.369 |
| Doc-first with an unlimited cache (simulated) | 0.365 | 0.491 |
| **nDCG@10** (query-first / doc-first / FP8) | 0.7561 / 0.7507 / 0.7476 | 0.3490 / 0.3549 / 0.3543 |

What we found:

- **Documents repeat, but most reuse is lost before it pays off.** LRU eviction and 64-token block alignment leave doc-first BF16 with 44% (SciFact) and 58% (NFCorpus) of the reuse an unlimited cache would give. On SciFact, query-first actually serves more tokens from cache. Its first 64-token block (system prompt + instruction + the start of the query) is shared by all 20 documents of a query, while doc-first's 59-token shared prefix is shorter than one block and never hits.
- **KV compression is a cache-capacity lever.** FP8 KV nearly doubles capacity, which raises doc-first's hit rate 1.41x / 1.29x and puts it ahead of query-first on both datasets. The cost is −0.0031 nDCG@10 on SciFact (95% CI [−0.0069, −0.0002]) and −0.0006 on NFCorpus vs vLLM BF16, both within one point. Without calibrated scales, vLLM's FP8 KV is about 5x noisier than BF16 (post-RoPE keys reach 488, above e4m3's 448). Our calibrated FP8 in Phase 5 cost −0.0013 / −0.0006.
- **There's a capacity cliff.** Rerunning trace 1 (840 SciFact requests, ~281k tokens) reuses 0.1% of tokens with BF16 KV and 91% with FP8. NFCorpus trace 1 (~165k tokens) fits in both (94%).
- **The behavior is predictable.** A block-level LRU simulator, given only the trace, the block size and the cache capacity, matches every measured hit rate to three decimals (18 setting × dataset × trace cases, cold and warm). Capacity planning for this workload can be done offline.
- **At 0.6B, cache hits don't change latency.** At concurrency 1, per-request latency is about 15 ms whether 0% or 94% of the prompt is cached. A linear fit over 31,640 requests per setting gives −2.0 to +1.5 ms per 1,000 uncached tokens (|r| ≤ 0.08). Fixed per-request overhead dominates prefill at this model size, so reuse would show up in latency only for larger models or longer documents. Throughput at concurrency 16 varied by up to ~30% between runs at equal hit rates, so we don't rank settings by it.

Traces 1 (same query, 20 docs) and 2 (same doc, 20 queries), concurrency 1/4/16 and warm reruns are in `results/vllm/table3.csv`; everything is summarized in `results/metrics/vllm.json`.

**Deviation from the pre-registration:** the prefix-cache gate (G3) was written for 16-token blocks. vLLM 0.34 chose 64, so G3 fails as written. At the actual block size it passes exactly on all 10 probes (cold request 0 tokens cached; second request exactly the 64-aligned document prefix).

## Limitations

- **Truncation:** docs are cut at 256 tokens (71 to 77% are truncated), so absolute scores sit below full-length published numbers. Rerank depth is 20.
- **Scope:** one model (0.6B), two datasets, one seed.
- **The order gate misses by 0.0013 on SciFact** (see above). The LoRA was trained on short MS MARCO passages (median 77 tokens) and evaluated on longer BEIR docs.
- **Fake quant:** values are dequantized to bf16 before attention. Quality and storage are measured; quantized-attention speed is not.
- **Post-RoPE keys:** keys are quantized after RoPE, because that's what is stored. Pre-RoPE quantization (KVQuant) may do better.
- **Strict margin:** the ±0.002 margin is about 10x the measured noise. INT4 KIVI lands in "within 1 point" because its CI is wider than the margin (on SciFact, on the upside), not because of a measured loss.
- **Shared prefix:** quality was measured with each doc's own copy of the shared prefix. Using one canonical copy, as the storage numbers assume, shifts scores by at most 0.048 log-odds (6x below the noise floor).
- **Pruning:** importance comes only from the document's own attention (the query is never seen, by design), averaged over all heads and layers into one kept set shared by every layer. Learned or per-layer selection may do better. Random pruning uses one seed. The early-token finding is post hoc. Pruning storage is measured on packed BF16 files; combined pruning + quantization comes in Phase 7.
- **Hardware:** Mac timing comes from a fanless laptop and is only relative.
- **Serving:** one rented RTX 4090, one run per configuration, vLLM 0.34 defaults (64-token blocks), a closed-loop replay of BEIR test queries with no arrival times, and FP8 KV without calibrated scales. The query-first serving baseline is the original model; the doc-first settings use the LoRA. These are benchmarks, not a production deployment.

## Roadmap

- **Phase 7:** uniform 75% pruning combined with FP8 and INT4 KIVI. Do the losses compound?
- **Phase 8b:** cold-disk load benchmarks for the packed formats, a length-matched LoRA retrain, and optional serving follow-ups (16-token blocks, calibrated FP8 KV scales).
- **Phase 9:** plots, final README, LoRA weights on Hugging Face.
- **Follow-ups:** a pre-registered "keep the first N tokens" pruning arm, learned or per-layer token selection.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e . --no-deps
python scripts/00_download_models.py && python scripts/00_check_env.py
python scripts/01_build_candidates.py
python scripts/02_tokenize.py && python scripts/02b_tokenize_docfirst.py
python scripts/03_score_reference.py --models qwen3 bge
python scripts/04_docfirst_zeroshot.py && python scripts/04b_random_baseline.py
python scripts/05_equality_test.py
python scripts/06a_build_train_data.py
python scripts/06_train_lora.py --order df --grad-ckpt --micro-groups 8   # GPU (RTX 4090), trains and merges
python scripts/06_train_lora.py --order qf --grad-ckpt --micro-groups 8
python scripts/06b_merge_check.py && python scripts/06c_eval_lora.py && python scripts/06d_equality_lora.py
python scripts/07a_noise_floor.py && python scripts/07_quant_sweep.py
python scripts/08_pack_storage.py
python scripts/09a_importance.py && python scripts/09_prune_sweep.py && python scripts/09b_prune_summary.py   # pruning (~1.7 h on the Air)
python scripts/11a_build_traces.py && python scripts/11c_gate_reference.py   # serving traces + gate reference
```

Serving (GPU, separate venv because vLLM pins its own torch), one setting at a time (`qf`, `df`, `df_fp8`):

```bash
python -m venv vllm-venv && source vllm-venv/bin/activate
pip install -r requirements-vllm.txt && pip install -e . --no-deps
bash scripts/11_serve.sh df
python scripts/11b_vllm_gate.py --setting df
python scripts/11_vllm_bench.py --setting df --traces t3 t1 t2 --concurrency 16 4 1
```

Summaries from the committed run files (the pruning summary also needs `09a`'s importance artifacts):

```bash
tar xzf results/vllm/runs.tar.gz -C results/vllm
python scripts/11d_vllm_summary.py
tar xzf results/scores/p6_scores.tar.gz
python scripts/09b_prune_summary.py
```

Every script writes JSON to `results/` with provenance (git commit, package versions, input hashes). Long runs are resumable and have a `--smoke N` mode. The LoRA weights will be on Hugging Face in Phase 9; until then, step 06 needs a GPU (about 1.5 h on an RTX 4090).

Tests: `python tests/test_<name>.py` (plain asserts).

## Repo layout

```
src/marginkv/   config, data, retrieval (BM25), models (Qwen3, bge), kv (cache ops, sweep, quant, pack, prune), serving (vLLM client), train (LoRA), eval
scripts/        numbered pipeline steps
tests/          parity, equality, quantizer, packing and pruning tests
results/        committed JSON: candidates, scores, metrics, traces, vLLM runs
```

## Credits

Builds directly on HyperRAG (An, Cheng, Park, Jiang, 2025) and the LMCache line of work on KV reuse. Quantization follows KIVI (Liu et al., 2024). Pruning baselines follow H2O (Zhang et al., 2023) and StreamingLLM (Xiao et al., 2023). The training recipe follows RankLLaMA (Ma et al., 2023). Serving uses [vLLM](https://github.com/vllm-project/vllm). Model: [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B). Data: [BEIR](https://github.com/beir-cellar/beir).

## License

MIT