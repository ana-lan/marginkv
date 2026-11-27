// app/app/layout.tsx
import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "MarginKV results explorer",
  description:
    "Storage vs. ranking quality for reusable, compressed document KV caches in a decoder-only reranker.",
};

// Runs before first paint: saved choice, else the system setting. Avoids a light flash in dark mode.
const THEME_SCRIPT = `(function(){var d=document.documentElement,t=null;try{t=localStorage.getItem("theme")}catch(e){}if(t!=="light"&&t!=="dark"){t=window.matchMedia&&matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light"}d.dataset.theme=t})();`;

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body>{children}</body>
    </html>
  );
}