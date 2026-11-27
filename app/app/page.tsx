// app/app/page.tsx  (placeholder; replaced in 9b.2)
import overview from "@/data/overview.json";

export default function Home() {
  const df = overview.systems.find((s) => s.id === "lora_dfB")!;
  return (
    <main className="wrap">
      <h1>MarginKV</h1>
      <p>Results explorer coming soon.</p>
      <p className="muted">
        Doc-first LoRA nDCG@10 on SciFact: {df.by_ds.scifact.ndcg10.toFixed(4)} (data export{" "}
        {overview.provenance.export_git_commit.slice(0, 7)})
      </p>
    </main>
  );
}