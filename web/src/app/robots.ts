import type { MetadataRoute } from "next";
import { SITE_ORIGIN, SITE_URL } from "@/lib/site";

// ``output: export`` (in next.config.ts) needs every route to be
// explicitly static; pin ``dynamic = "force-static"`` so the static
// export collector treats robots.txt as a build-time artifact.
export const dynamic = "force-static";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [{ userAgent: "*", allow: "/" }],
    sitemap: `${SITE_URL}/sitemap.xml`,
    host: SITE_ORIGIN,
  };
}
