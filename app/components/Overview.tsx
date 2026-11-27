// app/components/Overview.tsx
import overview from "@/data/overview.json";
import quant from "@/data/quant.json";
import trace3 from "@/data/trace3.json";
import { DS, REPO, ci, fx, pct, signed, times } from "@/lib/format";

export default function Overview() {
  const sys = Object.fromEntries(overview.systems.map((s) => [s.id, s]));
  const label = Object.fromEntries(overview.datasets.map((d) => [d.id, d.label]));
  const kivi = quant.formats.find((f) => f.id === "int4_kivi")!;
  const t = trace3.datasets;

  return (
    <section id="overview">
      <div className="row">
        <div>
          <p className="lede">
            A reranker reads every candidate document again for every query. MarginKV makes{" "}
            <b>Qwen3-Reranker-0.6B</b> read the document <i>first</i>, so the document&apos;s KV cache is built
            once and reused across queries, then measures how far that stored cache can be compressed before
            ranking quality drops, and how much of it a real serving stack actually reuses.
          </p>
          <p>
            Setup: BEIR SciFact and NFCorpus, BM25 top 100, top 20 reranked, documents truncated to 256 tokens,
            nDCG@10 and MRR@10 with paired bootstrap CIs. Decision rules were committed before the runs.
          </p>
        </div>
        <aside className="note">
          The name: the reranker reads each document once and keeps its notes (the KV cache) in the margins,
          compressed, for every later query.
        </aside>
      </div>

      <div className="tiles">
        <div className="tile">
          <div className="tile-k">Document first, quality kept</div>
          {DS.map((ds) => {
            const c = overview.dfB_vs_orig_qf[ds];
            return (
              <div key={ds} className="tile-line">
                <span>{label[ds]}</span>
                <span className="num">{signed(c.d_ndcg10)}</span>
                <span className="num dim">{ci(c.ndcg10_ci)}</span>
              </div>
            );
          })}
          <div className="tile-foot">nDCG@10, doc-first LoRA vs the original query-first model</div>
        </div>

        <div className="tile">
          <div className="tile-k">
            INT4 KIVI: <span className="num">{times(kivi.by_ds.scifact.ratio_vs_bf16)}</span> smaller
          </div>
          {DS.map((ds) => (
            <div key={ds} className="tile-line">
              <span>{label[ds]}</span>
              <span className="num">{signed(kivi.by_ds[ds].d_ndcg10)}</span>
              <span className="num dim">{kivi.by_ds[ds].mib_per_doc.toFixed(2)} MiB/doc</span>
            </div>
          ))}
          <div className="tile-foot">nDCG@10 change vs BF16 cache, packed bytes on disk</div>
        </div>

        <div className="tile">
          <div className="tile-k">Repeats are not hits</div>
          {DS.map((ds) => (
            <div key={ds} className="tile-line">
              <span>{label[ds]}</span>
              <span className="num">{pct(t[ds].repeat_frac_id, 0)}</span>
              <span className="num dim">
                {pct(t[ds].by_setting.df.hit_frac, 0)} / {pct(t[ds].by_setting.df_fp8.hit_frac, 0)}
              </span>
            </div>
          ))}
          <div className="tile-foot">
            docs that repeat in the query stream / prompt tokens from cache, BF16 / FP8 KV (vLLM)
          </div>
        </div>
      </div>

      <h2>Table 1: does document-first ordering cost quality?</h2>
      <div className="row">
        <div className="scroll">
          <table>
            <thead>
              <tr>
                <th>System</th>
                {DS.map((ds) => (
                  <th key={ds} className="r">
                    {label[ds]} nDCG@10 / MRR@10
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {overview.systems.map((s) => (
                <tr key={s.id} className={s.id === "lora_dfB" ? "hl" : undefined}>
                  <td>
                    {s.label}
                    <div className="sub">{s.note}</div>
                  </td>
                  {DS.map((ds) => (
                    <td key={ds} className="r num">
                      {fx(s.by_ds[ds].ndcg10)} / {fx(s.by_ds[ds].mrr10)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <aside className="note">
          Pre-registered order gate: doc-first LoRA within {overview.order_gate.scifact.threshold.toFixed(2)}{" "}
          nDCG@10 of the query-first LoRA control.{" "}
          {DS.map((ds) => {
            const g = overview.order_gate[ds];
            return (
              <span key={ds}>
                {label[ds]}: {signed(g.d_ndcg10)} {ci(g.ndcg10_ci)},{" "}
                {g.pass ? <b>pass</b> : <b>miss by {fx(-(g.d_ndcg10 + g.threshold))}</b>}.{" "}
              </span>
            );
          })}
          Reported as measured. Zero-shot doc-first ({fx(sys.qwen3_dfB_zs.by_ds.scifact.ndcg10)} on SciFact) is
          why the LoRA exists.
        </aside>
      </div>

      <p className="links">
        <a href={REPO}>Code and README on GitHub</a>
        <span aria-hidden> · </span>
        <a href={`${REPO}#results`}>Full results</a>
      </p>
    </section>
  );
}