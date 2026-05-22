import type { MetadataRoute } from "next";

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
