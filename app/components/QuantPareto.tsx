// app/components/QuantPareto.tsx
"use client";

import { useState } from "react";
import {
  CartesianGrid,
  ErrorBar,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import quant from "@/data/quant.json";
import { DS, type Ds, ci, fx, signed, times } from "@/lib/format";
import { useCssVars } from "@/lib/useCssVars";

const ONE_POINT = 0.01; // plan definition: "within 1 point" = within 0.01 nDCG@10
const VARS = ["ink", "muted", "rule", "accent", "warn", "bad", "surface"] as const;
const LABEL: Record<Ds, string> = { scifact: "SciFact", nfcorpus: "NFCorpus" };
const VERDICT_VAR = {
  baseline: "ink",
  no_measurable_loss: "accent",
  within_1_point: "warn",
  loss: "bad",
} as const;
type Verdict = keyof typeof VERDICT_VAR;
const VLABEL = quant.verdict_labels as Record<Verdict, string>;

type Row = {
  id: string;
  label: string;
  note: string;
  headline: boolean;
  verdict: Verdict;
  x: number;
  y: number;
  err: [number, number];
  lo: number;
  hi: number;
  ndcg10: number;
  mrr10: number;
  d_mrr10: number;
  ratio: number;
  naive: number;
};

function rowsFor(ds: Ds): Row[] {
  return quant.formats.map((f) => {
    const r = f.by_ds[ds];
    const [lo, hi] = r.ndcg10_ci;
    return {
      id: f.id,
      label: f.label,
      note: f.note,
      headline: f.headline,
      verdict: r.verdict as Verdict,
      x: r.mib_per_doc,
      y: r.d_ndcg10,
      err: [r.d_ndcg10 - lo, hi - r.d_ndcg10],
      lo,
      hi,
      ndcg10: r.ndcg10,
      mrr10: r.mrr10,
      d_mrr10: r.d_mrr10,
      ratio: r.ratio_vs_bf16,
      naive: r.ratio_vs_naive_bf16,
    };
  });
}

// Round axis: pick the smallest step that gives at most maxTicks ticks, snap ends to it.
function niceAxis(lo: number, hi: number, maxTicks = 7) {
  const steps = [0.001, 0.002, 0.0025, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1];
  for (const s of steps) {
    const a = Math.floor(lo / s) * s;
    const b = Math.ceil(hi / s) * s;
    const n = Math.round((b - a) / s) + 1;
    if (n <= maxTicks) {
      const ticks = Array.from({ length: n }, (_, i) => +(a + i * s).toFixed(6));
      const decimals = String(s).split(".")[1]?.length ?? 0;
      return { domain: [ticks[0], ticks[n - 1]] as [number, number], ticks, decimals };
    }
  }
  return { domain: [lo, hi] as [number, number], ticks: undefined, decimals: 3 };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function Tip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const r: Row = payload[0].payload;
  return (
    <div className="tip">
      <div className="tip-h">{r.label}</div>
      <div className="dim">{r.note}</div>
      <table>
      <tbody>
          <tr><td>nDCG@10</td><td className="num">{fx(r.ndcg10)}</td></tr>
          {r.verdict !== "baseline" && (
            <tr><td>Δ vs BF16</td><td className="num">{signed(r.y)} {ci([r.lo, r.hi])}</td></tr>
          )}
          <tr>
            <td>MRR@10</td>
            <td className="num">
              {fx(r.mrr10)}
              {r.verdict !== "baseline" && ` (${signed(r.d_mrr10)})`}
            </td>
          </tr>
          <tr><td>Packed size</td><td className="num">{r.x.toFixed(2)} MiB/doc</td></tr>
          {r.verdict !== "baseline" && (
            <tr><td>vs BF16</td><td className="num">{times(r.ratio)} smaller</td></tr>
          )}
          <tr><td>vs naive BF16</td><td className="num">{times(r.naive)} smaller</td></tr>
          <tr><td>Verdict</td><td><span className={`pill v-${r.verdict}`}>{VLABEL[r.verdict]}</span></td></tr>
        </tbody>
      </table>
    </div>
  );
}

export default function QuantPareto() {
  const [ds, setDs] = useState<Ds>("scifact");
  const [zoom, setZoom] = useState(false);
  const c = useCssVars(VARS);
  const rows = rowsFor(ds);
  const margin = quant.margin[ds];

  const xMax = Math.ceil(Math.max(...rows.map((r) => r.x)) / 5) * 5;
  const rawTop = Math.max(margin, ...rows.map((r) => r.hi)) + 0.001;
  const rawBottom = zoom ? -ONE_POINT - 0.002 : Math.min(...rows.map((r) => r.lo)) - 0.002;
  const axis = niceAxis(rawBottom, rawTop);
  const hidden = rows.filter((r) => r.lo < axis.domain[0]);
  const kivi = rows.find((r) => r.id === "int4_kivi")!;
  const tok = rows.find((r) => r.id === "int4_tok")!;

  return (
    <section id="quant">
      <h2>Storage vs. quality: quantizing the stored cache</h2>
      <div className="row">
        <div>
          <p>
            Each document&apos;s KV cache is stored once and reused for every query, so its size on disk is the cost.
            Each point is one format: packed size per document (real files: codes, scales, headers) against the
            change in nDCG@10 relative to the uncompressed BF16 cache, with a 95% paired bootstrap CI.
          </p>

          <div className="controls">
            <div className="seg" role="group" aria-label="Dataset">
              {DS.map((d) => (
                <button key={d} type="button" aria-pressed={d === ds} onClick={() => setDs(d)}>
                  {LABEL[d]}
                </button>
              ))}
            </div>
            <button type="button" className="plain-btn" aria-pressed={zoom} onClick={() => setZoom(!zoom)}>
              {zoom ? "Show all formats" : "Zoom to ±1 point"}
            </button>
          </div>

          <div className="chart" role="img" aria-label={`nDCG@10 change vs packed size per format, ${LABEL[ds]}`}>
            {c ? (
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 12, right: 16, bottom: 28, left: 8 }}>
                  <CartesianGrid stroke={c.rule} strokeDasharray="2 4" />
                  <XAxis
                    type="number"
                    dataKey="x"
                    domain={[0, xMax]}
                    tick={{ fill: c.muted, fontSize: 12 }}
                    stroke={c.rule}
                    label={{ value: "Packed MiB per document", position: "insideBottom", offset: -16, fill: c.muted, fontSize: 12 }}
                  />
                  <YAxis
                    type="number"
                    dataKey="y"
                    domain={axis.domain}
                    ticks={axis.ticks}
                    allowDataOverflow
                    tickFormatter={(v: number) => (Math.abs(v) < 1e-9 ? "0" : signed(v, axis.decimals))}
                    tick={{ fill: c.muted, fontSize: 12 }}
                    stroke={c.rule}
                    width={64}
                  />
                  <ReferenceArea y1={-margin} y2={margin} fill={c.accent} fillOpacity={0.14} stroke="none" />
                  <ReferenceLine y={0} stroke={c.muted} />
                  <ReferenceLine
                    y={-ONE_POINT}
                    stroke={c.warn}
                    strokeDasharray="5 4"
                    label={{ value: "−1 point", position: "insideBottomRight", fill: c.warn, fontSize: 12 }}
                  />
                  <Tooltip content={<Tip />} cursor={{ stroke: c.rule }} />
                  <Scatter
                    data={rows}
                    isAnimationActive={false}
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    shape={(p: any) => {
                      const r: Row = p.payload;
                      const col = c[VERDICT_VAR[r.verdict]];
                      return (
                        <g>
                          {r.headline && <circle cx={p.cx} cy={p.cy} r={10} fill="none" stroke={col} strokeWidth={1.5} />}
                          <circle cx={p.cx} cy={p.cy} r={r.headline ? 6 : 5} fill={col} stroke={c.surface} strokeWidth={1} />
                        </g>
                      );
                    }}
                  >
                    <ErrorBar dataKey="err" direction="y" width={6} stroke={c.muted} strokeWidth={1.2} />
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            ) : null}
          </div>

          <div className="legend">
            {(Object.keys(VERDICT_VAR) as Verdict[]).map((v) => (
              <span key={v}>
                <i className={`dot v-${v}`} /> {VLABEL[v]}
              </span>
            ))}
            <span>
              <i className="band" /> ±{margin} margin
            </span>
            <span className="dim">ringed: headline INT4</span>
          </div>
          {hidden.length > 0 && (
            <p className="dim small">
              Below this view:{" "}
              {hidden.map((r) => `${r.label} (${signed(r.y)} ${ci([r.lo, r.hi])})`).join(", ")}.
            </p>
          )}
        </div>

        <aside className="note">
          <b>Verdicts</b> (pre-registered): <i>no measurable loss</i> if the CI sits inside ±{margin};{" "}
          <i>within 1 point</i> if Δ ≥ −{ONE_POINT}; otherwise <i>loss</i>. The ±{margin} margin is the floor
          of the rule; measured bf16 run-to-run noise sits well inside it.
          <br />
          <br />
          Same 4 bits, different axis: {kivi.label} is {signed(kivi.y)}, {tok.label} is {signed(tok.y)} on{" "}
          {LABEL[ds]}. Post-RoPE keys have outlier channels, so keys need per-channel scales.
        </aside>
      </div>

      <div className="scroll">
        <table>
          <thead>
            <tr>
              <th>Format</th>
              <th className="r">MiB/doc</th>
              <th className="r">vs BF16</th>
              {DS.map((d) => (
                <th key={d} className="r">
                  {LABEL[d]} Δ nDCG@10 [95% CI]
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {quant.formats.map((f) => (
              <tr key={f.id} className={f.headline ? "hl" : undefined}>
                <td>
                  {f.label}
                  <div className="sub">{f.note}</div>
                </td>
                <td className="r num">{f.by_ds.scifact.mib_per_doc.toFixed(2)}</td>
                <td className="r num">{times(f.by_ds.scifact.ratio_vs_bf16)}</td>
                {DS.map((d) => {
                  const r = f.by_ds[d];
                  return (
                    <td key={d} className="r">
                      {r.verdict === "baseline" ? (
                        <span className="num">{fx(r.ndcg10)}</span>
                      ) : (
                        <>
                          <span className="num">
                            {signed(r.d_ndcg10)} <span className="dim">{ci(r.ndcg10_ci)}</span>
                          </span>
                          <div>
                            <span className={`pill v-${r.verdict}`}>{VLABEL[r.verdict as Verdict]}</span>
                          </div>
                        </>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="dim small">
        MiB/doc and ratios are SciFact means (NFCorpus agrees to 3 decimals). BF16 row shows absolute nDCG@10. Shared
        system and instruction prefix is stored once per corpus and not counted per document.
      </p>
    </section>
  );
}