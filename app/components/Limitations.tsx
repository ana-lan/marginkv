// app/components/Limitations.tsx
import trace3 from "@/data/trace3.json";

export default function Limitations() {
  return (
    <section id="limitations">
      <h2>Limitations</h2>
      <ul className="lim">
        <li>One model (Qwen3-Reranker-0.6B), two BEIR datasets, one seed.</li>
        <li>
          Documents are truncated to 256 tokens and only the BM25 top 20 is reranked, so absolute scores sit below
          published full-length numbers.
        </li>
        <li>
          Every doc-first and compression result depends on the LoRA adaptation. The doc-first LoRA misses the
          pre-registered order gate on SciFact by a small margin (CI includes zero); it is reported as is.
        </li>
        <li>
          Quantization quality comes from fake quantization (quantize, dequantize, score). Quality and stored bytes
          are measured; attention speed on compressed caches is not. Keys are quantized after RoPE. The shared
          system and instruction prefix stays BF16 and is stored once.
        </li>
        <li>
          Quality was measured with each document&apos;s own copy of the shared prefix; storage assumes one shared
          copy. The score effect is several times below bf16 noise.
        </li>
        <li>
          Serving numbers come from one rented RTX 4090, vLLM {trace3.vllm_version} defaults (
          {trace3.block_size}-token blocks), one run per configuration, replaying BEIR test queries in file order
          with no arrival times. It is a benchmark, not a production deployment. High-concurrency throughput is
          noisy run to run, so settings are not ranked on it.
        </li>
        <li>
          vLLM&apos;s FP8 KV used default scales (no calibration); its score error is larger than the calibrated FP8
          in the quantization study.
        </li>
        <li>
          The query-first serving baseline is the original model; the doc-first settings use the LoRA. Each is
          compared to its own Hugging Face reference.
        </li>
        <li>
          This is v0: it covers quantization and serving. Pruning, combined compression and disk load times are
          coming.
        </li>
      </ul>
    </section>
  );
}