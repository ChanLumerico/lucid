import type { NextConfig } from "next";
import path from "path";

const isProd = process.env.NODE_ENV === "production";
const basePath = isProd ? "/lucid" : "";

const nextConfig: NextConfig = {
  output: "export",
  basePath,
  assetPrefix: basePath,
  images: { unoptimized: true },
  turbopack: {
    root: path.resolve(__dirname),
  },
};

// Opt-in bundle analyzer.  Run ``ANALYZE=1 pnpm build`` to get an
// HTML report under ``.next/analyze/``.  We import lazily to avoid
// shipping the analyzer code into normal builds and skip the wrap
// when the package isn't installed — keeps the analyzer optional.
async function withOptionalAnalyzer(config: NextConfig): Promise<NextConfig> {
  if (process.env.ANALYZE !== "1") return config;
  try {
    // Untyped because we don't carry ``@next/bundle-analyzer`` as a
    // declared dep — it's optional-on-demand.  Cast through ``unknown``
    // so tsc doesn't fail builds in the (default) case where the
    // package isn't installed.
    const mod = (await import(
      /* @vite-ignore */ "@next/bundle-analyzer" as string
    )) as unknown as {
      default: (opts: { enabled: boolean; openAnalyzer?: boolean }) => (cfg: NextConfig) => NextConfig;
    };
    const withBundleAnalyzer = mod.default({ enabled: true, openAnalyzer: false });
    return withBundleAnalyzer(config);
  } catch {
    // Analyzer not installed — print a hint but don't fail the build.
    // eslint-disable-next-line no-console
    console.warn(
      "[next.config] ANALYZE=1 set but @next/bundle-analyzer not installed.\n"
      + "  Install: pnpm add -D @next/bundle-analyzer",
    );
    return config;
  }
}

export default withOptionalAnalyzer(nextConfig);
