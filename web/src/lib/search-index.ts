import fs from "fs";
import path from "path";
import type { ApiModule, ApiClass, ApiFunction, ApiMethod, ApiData, ApiClassModule } from "./types";
import { getAllDocMeta } from "./mdx-compile";

export interface SearchEntry {
  id: string;
  title: string;
  summary: string;
  href: string;
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

const MODULE_SLUGS = [
  "lucid",
  "lucid.tensor",
  "lucid.nn",
  "lucid.nn.functional",
  "lucid.nn.init",
  "lucid.nn.utils",
  "lucid.optim",
  "lucid.autograd",
  "lucid.func",
  "lucid.linalg",
  "lucid.fft",
  "lucid.signal",
  "lucid.special",
  "lucid.distributions",
  "lucid.utils.data",
  "lucid.amp",
  "lucid.profiler",
  "lucid.einops",
  "lucid.serialization",
];

function collectMethods(
  cls: ApiClass | ApiClassModule,
  entries: SearchEntry[],
) {
  for (const method of cls.methods ?? []) {
    const m = method as ApiMethod;
    entries.push({
      id: `${cls.path}.${m.name}`,
      title: `${cls.name}.${m.name}`,
      summary: m.summary ?? "",
      href: `/api/${cls.path}#${m.name}`,
      kind: "api-function",
      badge: cls.name,
    });
  }
}

function collectMembers(mod: ApiModule, entries: SearchEntry[]) {
  for (const member of mod.members ?? []) {
    if (member.kind === "class") {
      const cls = member as ApiClass;
      entries.push({
        id: cls.path,
        title: cls.name,
        summary: cls.summary ?? "",
        href: `/api/${cls.path}`,
        kind: "api-class",
        badge: cls.path.split(".").slice(0, -1).join("."),
      });
      collectMethods(cls, entries);
    } else if (member.kind === "function") {
      const fn = member as ApiFunction;
      entries.push({
        id: fn.path,
        title: fn.name,
        summary: fn.summary ?? "",
        href: `/api/${fn.path.split(".").slice(0, -1).join(".")}#${fn.name}`,
        kind: "api-function",
        badge: fn.path.split(".").slice(0, -1).join("."),
      });
    }
  }
}

export function buildSearchIndex(): SearchEntry[] {
  const entries: SearchEntry[] = [];

  // API entries from Griffe JSON
  for (const slug of MODULE_SLUGS) {
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
      collectMembers(data as ApiModule, entries);
    } else if (data.kind === "class-module") {
      collectMethods(data as ApiClassModule, entries);
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
