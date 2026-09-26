import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone", // self-contained server for the VM: node .next/standalone/server.js
  poweredByHeader: false, // no X-Powered-By: Next.js
  devIndicators: { position: "bottom-right" },
};

export default nextConfig;
