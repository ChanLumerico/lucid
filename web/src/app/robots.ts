import type { MetadataRoute } from "next";

// ``output: export`` (in next.config.ts) needs every route to be
// explicitly static; pin ``dynamic = "force-static"`` so the static
// export collector treats robots.txt as a build-time artifact.
export const dynamic = "force-static";

const ORIGIN =
  process.env.LUCID_DOCS_ORIGIN ??
  process.env.NEXT_PUBLIC_LUCID_DOCS_ORIGIN ??
  "https://lucid.docs.local";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [{ userAgent: "*", allow: "/" }],
    sitemap: `${ORIGIN}/sitemap.xml`,
    host: ORIGIN,
  };
}
