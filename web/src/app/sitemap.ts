import type { MetadataRoute } from "next";
import { getAllModuleSlugs, loadApiData } from "@/lib/api-loader";
import { isApiModule, isApiClassModule } from "@/lib/types";
import { getAllDocSlugs } from "@/lib/mdx-compile";

// ``output: export`` (in next.config.ts) needs every route to be
// explicitly static; pin ``dynamic = "force-static"`` so the static
// export collector treats sitemap.xml as a build-time artifact.
export const dynamic = "force-static";

/** Site identity for the sitemap.  Lives in the build-meta hint when set
 *  (e.g. ``LUCID_DOCS_ORIGIN=https://lucid.docs.example.com``); falls
 *  back to a sensible default that crawlers can still discover via the
 *  ``<link rel="canonical">`` tags Next emits per page. */
const ORIGIN =
  process.env.LUCID_DOCS_ORIGIN ??
  process.env.NEXT_PUBLIC_LUCID_DOCS_ORIGIN ??
  "https://lucid.docs.local";

const now = new Date();

/** Static + dynamic route enumerator for Next.js's MetadataRoute.Sitemap
 *  contract.  Pulls module + member slugs from the same JSON the docs
 *  pages themselves load, so the sitemap can never drift from the
 *  actual published surface. */
export default function sitemap(): MetadataRoute.Sitemap {
  const entries: MetadataRoute.Sitemap = [
    { url: `${ORIGIN}/`,           lastModified: now, changeFrequency: "weekly",  priority: 1.0 },
    { url: `${ORIGIN}/api`,        lastModified: now, changeFrequency: "weekly",  priority: 0.9 },
    { url: `${ORIGIN}/docs`,       lastModified: now, changeFrequency: "weekly",  priority: 0.7 },
    { url: `${ORIGIN}/changelog`,  lastModified: now, changeFrequency: "monthly", priority: 0.4 },
  ];

  // API module + member pages.
  for (const slug of getAllModuleSlugs()) {
    entries.push({
      url: `${ORIGIN}/api/${slug}`,
      lastModified: now,
      changeFrequency: "weekly",
      priority: 0.6,
    });
    try {
      const data = loadApiData(slug);
      if (isApiModule(data)) {
        for (const m of data.members) {
          entries.push({
            url: `${ORIGIN}/api/${slug}/${m.name}`,
            lastModified: now,
            changeFrequency: "weekly",
            priority: 0.5,
          });
        }
      } else if (isApiClassModule(data)) {
        for (const meth of data.methods) {
          entries.push({
            url: `${ORIGIN}/api/${slug}/${meth.name}`,
            lastModified: now,
            changeFrequency: "weekly",
            priority: 0.5,
          });
        }
      }
    } catch {
      // module JSON missing — skip silently; the module page itself is
      // still in the sitemap so the crawler will find it.
    }
  }

  // MDX guides under /docs/[...slug].
  try {
    for (const slugParts of getAllDocSlugs()) {
      entries.push({
        url: `${ORIGIN}/docs/${slugParts.join("/")}`,
        lastModified: now,
        changeFrequency: "weekly",
        priority: 0.7,
      });
    }
  } catch {
    // content/ missing — fine.
  }

  return entries;
}
