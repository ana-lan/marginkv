// app/lib/useCssVars.ts
"use client";

import { useEffect, useState } from "react";

// Reads CSS custom properties from :root and re-reads them when data-theme changes,
// so SVG charts get real color values in both themes.
export function useCssVars<T extends string>(names: readonly T[]): Record<T, string> | null {
  const [vars, setVars] = useState<Record<T, string> | null>(null);

  useEffect(() => {
    const read = () => {
      const s = getComputedStyle(document.documentElement);
      return Object.fromEntries(names.map((n) => [n, s.getPropertyValue(`--${n}`).trim()])) as Record<T, string>;
    };
    setVars(read());
    const mo = new MutationObserver(() => setVars(read()));
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => mo.disconnect();
    // names is a module-level constant in every caller
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return vars;
}