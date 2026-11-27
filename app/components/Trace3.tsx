// app/components/Trace3.tsx
import storage from "@/data/storage.json";
import trace3 from "@/data/trace3.json";
import { DS, fx, pct, signed, times } from "@/lib/format";

const SET = Object.fromEntries(trace3.settings.map((s) => [s.id, s]));
const kTok = (x: number) => (x >= 1e6 ? `${(x / 1e6).toFixed(2)}M` : `${Math.round(x / 1000)}k`);
const both = (f: (ds: (typeof DS)[number]) => string) => DS.map(f).join(" / ");

export default function Trace3() {
  const D = trace3.datasets;
  const capBf16 = SET.df.kv_cache_tokens;
  const capFp8 = SET.df_fp8.kv_cache_tokens;
  const maxSimErr = Math.max(
    ...DS.flatMap((ds) => trace3.settings.map((s) => Math.abs(D[ds].by_setting[s.id as "qf"].sim_abs_err_c1_cold))),
  );
  const c1 = DS.flatMap((ds) => trace3.settings.map((s) => D[ds].by_setting[s.id as "qf"].c1_p50_ms));

  return (
    <section id="serving">
      <h2>Serving: does the stored cache actually get reused?</h2>
      <div className="row">
        <div>
          <p>
            The real BEIR query stream, replayed in order through vLLM {trace3.vllm_version} with prefix caching on one
            RTX 4090: every query in file order, each with its 20 BM25 candidates. Many documents come back for later
            queries. The question is how much of that a real cache turns into reuse.
          </p>

          <div className="bars2">
            {DS.map((ds) => {
              const d = D[ds];
              const s = d.by_setting;
              const items = [
                { k: "rep", label: "Documents that repeat in the stream", v: d.repeat_frac_id, cls: "b-ref" },
                { k: "unb", label: `Doc-first, unlimited cache (simulated, ${trace3.block_size}-token blocks)`, v: s.df.sim_unbounded_doc_frac, cls: "b-sim" },
                { k: "fp8", label: `Doc-first, FP8 KV (${kTok(capFp8)} tokens of cache)`, v: s.df_fp8.doc_tokens_reused_frac, cls: "b-fp8" },
                { k: "bf16", label: `Doc-first, BF16 KV (${kTok(capBf16)} tokens of cache)`, v: s.df.doc_tokens_reused_frac, cls: "b-bf16" },
                { k: "qf", label: "Query-first (original model)", v: s.qf.doc_tokens_reused_frac, cls: "b-qf" },
              ];
              return (
                <div key={ds} className="panel">
                  <div className="panel-h">
                    {d.label} <span className="dim small">{d.n_requests.toLocaleString("en-US")} requests</span>
                  </div>
                  {items.map((it) => (
                    <div key={it.k} className="hb">
                      <div className="hb-l">{it.label}</div>
                      <div className="hb-track">
                        <div className={`hb-fill ${it.cls}`} style={{ width: `${(it.v * 100).toFixed(1)}%` }} />
                      </div>
                      <div className="hb-v num">{pct(it.v, 0)}</div>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
          <p className="dim small">
            First bar: share of requests whose document appeared earlier in the stream. Other bars: share of document
            tokens served from the prefix cache (measured runs: cold cache, concurrency 16).
          </p>
        </div>

        <aside className="note">
        <b>Capacity.</b> A document comes back a median of {both((ds) => D[ds].reuse_distance_p50.toFixed(0))} requests
          later, about {both((ds) => kTok(D[ds].reuse_distance_p50 * D[ds].prompt_len_df_mean))} tokens of KV in between
          (p90: {both((ds) => kTok(D[ds].reuse_distance_p90 * D[ds].prompt_len_df_mean))}). The cache holds{" "}
          {kTok(capBf16)} tokens in BF16 and {kTok(capFp8)} in FP8.{" "}
          {DS.map((ds) => {
            const gap = D[ds].reuse_distance_p50 * D[ds].prompt_len_df_mean;
            const where =
              gap > capFp8
                ? "the typical gap is larger than both caches"
                : gap > capBf16
                  ? "the typical gap is larger than the BF16 cache but fits in FP8"
                  : "the typical gap fits in both, so the long tail is what gets evicted";
            return (
              <span key={ds}>
                {D[ds].label}: {where}.{" "}
              </span>
            );
          })}
          <br />
          <br />
          <b>Blocks.</b> vLLM matches prefixes in {trace3.block_size}-token blocks. The shared {storage.shared_len}-token
          prefix is shorter than one block, so doc-first never reuses it, while query-first&apos;s first block is shared by a
          query&apos;s 20 documents ({both((ds) => pct(D[ds].by_setting.qf.hit_frac, 0))} of prompt tokens from cache, almost
          no document reuse).
          <br />
          <br />
          <b>Compression as capacity.</b> FP8 KV holds {times(capFp8 / capBf16)} the tokens and lifts document reuse{" "}
          {both((ds) => times(D[ds].by_setting.df_fp8.doc_tokens_reused_frac / D[ds].by_setting.df.doc_tokens_reused_frac))}{" "}
          for {both((ds) => signed(D[ds].by_setting.df_fp8.ndcg10 - D[ds].by_setting.df.ndcg10))} nDCG@10.
          <br />
          <br />
          An offline block-level LRU simulator matched every measured hit rate
          {maxSimErr < 5e-4 ? " to 3 decimals" : ` (largest gap ${fx(maxSimErr, 3)})`}. At 0.6B, latency is per-request
          overhead: c=1 median {Math.min(...c1).toFixed(1)} to {Math.max(...c1).toFixed(1)} ms in every setting, hit or
          miss.
        </aside>
      </div>

      <div className="scroll">
        <table>
          <thead>
            <tr>
              <th>Setting</th>
              <th className="r">KV cache</th>
              {DS.map((ds) => (
                <th key={ds} className="r">
                  {D[ds].label}: prompt hit / doc reuse
                </th>
              ))}
              <th className="r">nDCG@10</th>
              <th className="r">c=1 p50 ms</th>
            </tr>
          </thead>
          <tbody>
            {trace3.settings.map((st) => {
              const id = st.id as "qf" | "df" | "df_fp8";
              return (
                <tr key={id} className={id === "df_fp8" ? "hl" : undefined}>
                  <td>{st.label}</td>
                  <td className="r num">{kTok(st.kv_cache_tokens)} tok</td>
                  {DS.map((ds) => (
                    <td key={ds} className="r num">
                      {pct(D[ds].by_setting[id].hit_frac)} / {pct(D[ds].by_setting[id].doc_tokens_reused_frac)}
                      {id !== "qf" && (
                        <div className="sub">
                          {pct(D[ds].by_setting[id].realized_vs_unbounded ?? 0, 0)} of unlimited cache
                        </div>
                      )}
                    </td>
                  ))}
                  <td className="r num">{both((ds) => fx(D[ds].by_setting[id].ndcg10))}</td>
                  <td className="r num">{both((ds) => D[ds].by_setting[id].c1_p50_ms.toFixed(1))}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="dim small">
        Prompt hit: share of all prompt tokens served from cache. Doc reuse: share of document tokens. nDCG@10 and latency
        as SciFact / NFCorpus. Query-first uses the original model, doc-first the LoRA; each was checked against its own
        Hugging Face reference. High-concurrency throughput was noisy run to run, so settings are not ranked on it.
      </p>
    </section>
  );
}