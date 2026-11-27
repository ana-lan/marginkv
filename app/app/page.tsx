// app/app/page.tsx
import Limitations from "@/components/Limitations";
import Overview from "@/components/Overview";
import Provenance from "@/components/Provenance";
import ThemeToggle from "@/components/ThemeToggle";
import QuantPareto from "@/components/QuantPareto";
import { REPO } from "@/lib/format";

const NAV = [
  { id: "overview", label: "Overview" },
  { id: "quant", label: "Quantization" },
  { id: "limitations", label: "Limitations" },
];

export default function Home() {
  return (
    <main className="wrap">
      <header className="top">
        <div className="brand">
          <h1>MarginKV</h1>
          <p className="tag">Storage-efficient KV cache reuse for decoder-only reranking</p>
        </div>
        <nav>
          {NAV.map((n) => (
            <a key={n.id} href={`#${n.id}`}>
              {n.label}
            </a>
          ))}
          <a href={REPO}>GitHub</a>
          <ThemeToggle />
        </nav>
      </header>
      <Overview />
      <QuantPareto />
      <Limitations />
      <Provenance />
    </main>
  );
}