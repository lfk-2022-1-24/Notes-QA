import type { NextConfig } from "next";

function parseAllowedDevOrigins(): string[] | undefined {
  const raw = process.env.NEXT_ALLOWED_DEV_ORIGINS;
  if (!raw) return undefined;
  const origins = raw
    .split(/[,\s]+/g)
    .map((s) => s.trim())
    .filter(Boolean);
  return origins.length ? origins : undefined;
}

const nextConfig: NextConfig = {
  serverExternalPackages: ["pg", "pdf-parse"],
  experimental: {
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },
  ...(process.env.NODE_ENV === "development"
    ? { allowedDevOrigins: parseAllowedDevOrigins() }
    : {}),
};

export default nextConfig;
