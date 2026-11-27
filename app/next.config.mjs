// app/next.config.mjs
import path from "node:path";
import { fileURLToPath } from "node:url";

const appDir = path.dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",              // static HTML in app/out, no server
  images: { unoptimized: true }, // next/image optimization needs a server
  reactStrictMode: true,
  turbopack: { root: appDir },   // ignore the stray lockfile in the home folder
};

export default nextConfig;