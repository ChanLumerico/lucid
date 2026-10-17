import type { MetadataRoute } from "next";
import { getAllModuleSlugs, loadApiData } from "@/lib/api-loader";
import { isApiModule, isApiClassModule } from "@/lib/types";
import { getAllDocSlugs } from "@/lib/mdx-compile";
import { getDiagrams } from "@/lib/architecture";
import { SITE_URL } from "@/lib/site";

// ``output: export`` (in next.config.ts) needs every route to be
// explicitly static; pin ``dynamic = "force-static"`` so the static
// export collector treats sitemap.xml as a build-time artifact.
export const dynamic = "force-static";

const now = new Date();

/** Static + dynamic route enumerator for Next.js's MetadataRoute.Sitemap
 *  contract.  Pulls module + member slugs from the same JSON the docs
 *  pages themselves load, so the sitemap can never drift from the
 *  actual published surface. */
export default function sitemap(): MetadataRoute.Sitemap {
  const entries: MetadataRoute.Sitemap = [
    { url: `${SITE_URL}/`,           lastModified: now, changeFrequency: "weekly",  priority: 1.0 },
    { url: `${SITE_URL}/api`,        lastModified: now, changeFrequency: "weekly",  priority: 0.9 },
    { url: `${SITE_URL}/docs`,       lastModified: now, changeFrequency: "weekly",  priority: 0.7 },
    { url: `${SITE_URL}/architecture`, lastModified: now, changeFrequency: "monthly", priority: 0.6 },
    { url: `${SITE_URL}/changelog`,  lastModified: now, changeFrequency: "monthly", priority: 0.4 },
  ];

  // Architecture diagram pages.
  for (const d of getDiagrams()) {
    entries.push({
      url: `${SITE_URL}/architecture/${d.slug}`,
      lastModified: now,
      changeFrequency: "monthly",
      priority: 0.5,
    });
  }

  // API module + member pages.
  for (const slug of getAllModuleSlugs()) {
    entries.push({
      url: `${SITE_URL}/api/${slug}`,
      lastModified: now,
      changeFrequency: "weekly",
      priority: 0.6,
    });
    try {
      const data = loadApiData(slug);
      if (isApiModule(data)) {
        for (const m of data.members) {
          entries.push({
            url: `${SITE_URL}/api/${slug}/${m.name}`,
            lastModified: now,
            changeFrequency: "weekly",
            priority: 0.5,
          });
        }
      } else if (isApiClassModule(data)) {
        for (const meth of data.methods) {
          entries.push({
            url: `${SITE_URL}/api/${slug}/${meth.name}`,
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
        url: `${SITE_URL}/docs/${slugParts.join("/")}`,
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
