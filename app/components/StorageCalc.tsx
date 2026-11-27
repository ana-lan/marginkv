// app/components/StorageCalc.tsx
"use client";

import { useState } from "react";
import quant from "@/data/quant.json";
import storage from "@/data/storage.json";
import { times } from "@/lib/format";

type Fit = (typeof storage.formats)[number];
const G = storage.group;
const BF16 = storage.formats.find((f) => f.id === "bf16")!;
const HEADLINE = new Set(quant.formats.filter((f) => f.headline).map((f) => f.id));
const NOTE = Object.fromEntries(quant.formats.map((f) => [f.id, f.note]));

// exact packed size model, fitted in scripts/14_export_app_data.py from the real file format
const size = (f: Fit, t: number) => f.a_bytes + f.b_bytes_per_token * t + f.c_bytes_per_group * Math.ceil(t / G);
const perToken = (f: Fit) => f.b_bytes_per_token + f.c_bytes_per_group / G;

function human(x: number, base: 1000 | 1024) {
  const units = base === 1000 ? ["B", "KB", "MB", "GB", "TB", "PB", "EB"] : ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
  let i = 0;
  while (x >= base && i < units.length - 1) {
    x /= base;
    i++;
  }
  const s = x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2);
  return `${s} ${units[i]}`;
}

// fit error label: tiny errors are exact up to header rounding
const errLabel = (e: number) =>
  Math.abs(e) < 5e-5 ? "exact" : `${e < 0 ? "\u2212" : "+"}${Math.abs(e * 100).toFixed(2)}%`;


type Preset = { id: string; label: string; n: number; t: number; measured?: Record<string, number>; err?: Record<string, number> };
const PRESETS: Preset[] = [
  ...storage.presets.map((p) => ({
    id: p.id,
    label: `${p.label} (${p.num_docs.toLocaleString("en-US")} docs)`,
    n: p.num_docs,
    t: Math.round(p.mean_doc_tokens * 100) / 100,
    measured: p.corpus_total_bytes as Record<string, number>,
    err: p.fit_rel_err as Record<string, number>,
  })),
  { id: "1m", label: "1M docs × 300 tokens", n: 1_000_000, t: 300 },
];

export default function StorageCalc() {
  const [nStr, setN] = useState(String(PRESETS[0].n));
  const [tStr, setT] = useState(String(PRESETS[0].t));
  const [active, setActive] = useState<string | null>(PRESETS[0].id);

  const n = Math.max(0, Math.floor(Number(nStr) || 0));
  const t = Math.max(0, Number(tStr) || 0);
  const ok = n > 0 && t > 0;
  const preset = PRESETS.find((p) => p.id === active);

  const naiveTotal = n * size(BF16, t + storage.shared_len);
  const rows = storage.formats.map((f) => {
    const perDoc = size(f, t);
    const total = storage.shared_file_bytes + n * perDoc;
    return { f, perDoc, total, ratio: naiveTotal / total };
  });

  function pick(p: Preset) {
    setN(String(p.n));
    setT(String(p.t));
    setActive(p.id);
  }

  return (
    <section id="storage">
      <h2>Storage calculator</h2>
      <div className="row">
        <div>
          <p>
            How much disk does a document KV store need? Pick a corpus size and an average document length. Sizes come
            from the exact packed file format (codes, scales, headers), with the shared system and instruction prefix
            stored once per corpus.
          </p>

          <div className="controls">
            {PRESETS.map((p) => (
              <button key={p.id} type="button" className="plain-btn" aria-pressed={active === p.id} onClick={() => pick(p)}>
                {p.label}
              </button>
            ))}
          </div>

          <div className="calc-in">
            <label>
              Documents
              <input
                type="number"
                min={1}
                step={1}
                value={nStr}
                onChange={(e) => {
                  setN(e.target.value);
                  setActive(null);
                }}
              />
              <span className="dim small">{ok ? n.toLocaleString("en-US") : ""}</span>
            </label>
            <label>
              Average tokens per document
              <input
                type="number"
                min={1}
                step="any"
                value={tStr}
                onChange={(e) => {
                  setT(e.target.value);
                  setActive(null);
                }}
              />
              <span className="dim small">after the shared {storage.shared_len}-token prefix</span>
            </label>
          </div>

          {ok ? (
            <div className="scroll">
              <table className="calc-table">
                <thead>
                  <tr>
                    <th>Format</th>
                    <th className="r">Per token</th>
                    <th className="r">Per doc</th>
                    <th className="r">Corpus total</th>
                    {preset?.measured && <th className="r">Measured files</th>}
                    <th className="r">vs naive</th>
                    <th style={{ width: "18%" }} />
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>
                      Naive BF16
                      <div className="sub">prefix repeated in every doc</div>
                    </td>
                    <td className="r num">{human(perToken(BF16), 1024)}</td>
                    <td className="r num">{human(size(BF16, t + storage.shared_len), 1000)}</td>
                    <td className="r num">
                      {human(naiveTotal, 1000)}
                      <div className="sub">{human(naiveTotal, 1024)}</div>
                    </td>
                    {preset?.measured && <td />}
                    <td className="r num">{times(1)}</td>
                    <td>
                      <div className="bar" style={{ width: "100%", opacity: 0.35 }} />
                    </td>
                  </tr>
                  {rows.map(({ f, perDoc, total, ratio }) => (
                    <tr key={f.id} className={HEADLINE.has(f.id) ? "hl" : undefined}>
                      <td>
                        {f.label}
                        <div className="sub">{NOTE[f.id]}</div>
                      </td>
                      <td className="r num">{human(perToken(f), 1024)}</td>
                      <td className="r num">{human(perDoc, 1000)}</td>
                      <td className="r num">
                        {human(total, 1000)}
                        <div className="sub">{human(total, 1024)}</div>
                      </td>
                      {preset?.measured && (
                        <td className="r num">
                          {human(preset.measured[f.id], 1024)}
                          <div className="sub">fit {errLabel(preset.err![f.id])}</div>
                        </td>
                      )}
                      <td className="r num">{times(ratio)}</td>
                      <td>
                        <div className="bar" style={{ width: `${(100 / ratio).toFixed(1)}%` }} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="dim">Enter a positive number of documents and tokens.</p>
          )}
        </div>

        <aside className="note">
          <b>Size model</b> per document: <span className="num">a + b·tokens + c·⌈tokens/{G}⌉</span>, fitted exactly to the
          packed format. The last term is INT4 KIVI&apos;s per-{G}-token key scale groups; it is zero for the others.
          <br />
          <br />
          Each corpus also stores one shared prefix file ({human(storage.shared_file_bytes, 1024)}). With a dataset
          preset, &quot;measured&quot; is the total from real packed files; the small gap comes from using the average
          length.
          <br />
          <br />
          Sizes hold for any length; ranking quality was measured on documents up to 256 tokens. Verdicts are in the
          chart above.
        </aside>
      </div>
    </section>
  );
}