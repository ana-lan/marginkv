// app/lib/format.ts
export const REPO = "https://github.com/ana-lan/marginkv";
export const DS = ["scifact", "nfcorpus"] as const;
export type Ds = (typeof DS)[number];

const MINUS = "\u2212";

export const fx = (x: number, d = 4) => (x < 0 ? MINUS : "") + Math.abs(x).toFixed(d);
export const signed = (x: number, d = 4) => (x < 0 ? MINUS : "+") + Math.abs(x).toFixed(d);
export const ci = (c: number[], d = 4) => `[${signed(c[0], d)}, ${signed(c[1], d)}]`;
export const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`;
export const times = (x: number, d = 2) => `${x.toFixed(d)}x`;
export const short = (sha: string | null | undefined) => (sha ? sha.slice(0, 7) : "unknown");