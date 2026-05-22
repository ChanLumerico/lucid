import fs from "fs";
import path from "path";
import type { ApiModule, ApiClass, ApiFunction, ApiMethod, ApiData, ApiClassModule } from "./types";
import { getAllDocMeta } from "./mdx-compile";
import { getAllModuleSlugs } from "./api-loader";

export interface SearchEntry {
  id: string;
  title: string;
  summary: string;
  href: string;
  /** Categorical hint that drives the icon + colour in the search
   *  dialog.  Engine symbols share the ``api-class`` / ``api-function``
   *  kinds so users see a consistent treatment — the ``badge`` field
   *  carries the slug context so ``lucid._C.engine`` results are still
   *  visually distinguishable. */
  kind: "api-module" | "api-class" | "api-function" | "doc";
  badge?: string;
}

function loadApiData(slug: string): ApiData | null {
  const file = path.join(
    process.cwd(),
    "public",
    "api-data",
    `${slug}.json`,
  );
  if (!fs.existsSync(file)) return null;
  return JSON.parse(fs.readFileSync(file, "utf-8")) as ApiData;
}

function collectMethods(
  slug: string,
  cls: ApiClass | ApiClassModule,
  entries: SearchEntry[],
) {
  for (const method of cls.methods ?? []) {
    const m = method as ApiMethod;
    entries.push({
      id: `${slug}/${cls.name}.${m.name}`,
      title: `${cls.name}.${m.name}`,
      summary: m.summary ?? "",
      // Anchor on the parent class's detail page so the URL works both
      // for Python (path-style) and C++ (engine slug + class name).
      href: `/api/${slug}/${cls.name}#${m.name}`,
      kind: "api-function",
      badge: cls.name,
    });
  }
}

function collectMembers(slug: string, mod: ApiModule, entries: SearchEntry[]) {
  for (const member of mod.members ?? []) {
    if (member.kind === "class") {
      const cls = member as ApiClass;
      entries.push({
        id: `${slug}/${cls.name}`,
        title: cls.name,
        summary: cls.summary ?? "",
        href: `/api/${slug}/${cls.name}`,
        kind: "api-class",
        badge: slug,
      });
      collectMethods(slug, cls, entries);
    } else if (member.kind === "function") {
      const fn = member as ApiFunction;
      entries.push({
        id: `${slug}/${fn.name}`,
        title: fn.name,
        summary: fn.summary ?? "",
        // Free functions live on the module page; anchor on their name.
        href: `/api/${slug}/${fn.name}`,
        kind: "api-function",
        badge: slug,
      });
    }
  }
}

export function buildSearchIndex(): SearchEntry[] {
  const entries: SearchEntry[] = [];

  // API entries from every JSON in public/api-data/.  Drives from
  // ``getAllModuleSlugs()`` (which already skips ``_*.json`` caches and
  // C++ per-class details) so adding a new module — Python or engine —
  // does not require touching this file.
  for (const slug of getAllModuleSlugs()) {
    const data = loadApiData(slug);
    if (!data) continue;

    entries.push({
      id: slug,
      title: slug,
      summary: data.summary ?? "",
      href: `/api/${slug}`,
      kind: "api-module",
    });

    if (data.kind === "module") {
      collectMembers(slug, data as ApiModule, entries);
    } else if (data.kind === "class-module") {
      collectMethods(slug, data as ApiClassModule, entries);
    }
  }

  // Doc entries from MDX frontmatter
  try {
    const docs = getAllDocMeta();
    for (const doc of docs) {
      entries.push({
        id: "doc:" + doc.slug,
        title: doc.title,
        summary: doc.description ?? "",
        href: doc.href,
        kind: "doc",
        badge: doc.category,
      });
    }
  } catch {
    // content/ may not exist yet
  }

  return entries;
}
