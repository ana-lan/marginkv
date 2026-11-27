// app/components/Provenance.tsx
import overview from "@/data/overview.json";
import quant from "@/data/quant.json";
import storage from "@/data/storage.json";
import trace3 from "@/data/trace3.json";
import { REPO, short } from "@/lib/format";

export default function Provenance() {
  const all = [overview, quant, storage, trace3];
  const p = overview.provenance;
  const seen = new Map<string, (typeof p.sources)[number]>();
  for (const d of all) for (const s of d.provenance.sources) seen.set(s.path, s);
  const commit = p.export_git_commit;

  return (
    <footer className="foot">
      <p>
        Every number on this page is read from committed results files by{" "}
        <a href={`${REPO}/blob/${commit}/scripts/14_export_app_data.py`}>scripts/14_export_app_data.py</a>{" "}
        (exported at commit <span className="num">{short(commit)}</span>
        {p.export_git_dirty ? ", with uncommitted changes" : ""}). No numbers are typed by hand.
      </p>
      <ul>
        {[...seen.values()].map((s) => (
          <li key={s.path}>
            <a href={`${REPO}/blob/${commit}/${s.path}`}>{s.path}</a>{" "}
            <span className="num dim">sha256 {s.sha256.slice(0, 12)}</span>
          </li>
        ))}
      </ul>
    </footer>
  );
}