<!-- README.md -->
# MarginKV

Storage-efficient KV cache reuse for decoder-only reranking.

A decoder-only reranker that reads each document once and keeps its KV cache "in the margins": compressed, stored, and reused for every query that retrieves that document.

**Status:** work in progress.

## Questions

1. Can Qwen3-Reranker-0.6B score with the document first, so its KV cache can be reused across queries, without losing ranking quality?
2. How far can that cached document KV be compressed (quantization, query-independent token pruning, or both) before ranking quality drops?

## Setup

BEIR SciFact and NFCorpus, BM25 top 100, rerank top 20, nDCG@10 and MRR@10. Builds on HyperRAG (An et al., 2025), which established document-side KV reuse for decoder-only rerankers.

Results, plots and reproduction steps will be added as each phase lands.

## License

MIT