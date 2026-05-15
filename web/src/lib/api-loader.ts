import { readFileSync } from "fs";
import { join } from "path";
import type {
  ApiData,
  ApiModule,
  ApiClassModule,
  ApiMember,
  ModuleSlug,
  MODULE_SLUGS,
} from "./types";

const API_DATA_DIR = join(process.cwd(), "public", "api-data");

// ---------------------------------------------------------------------------
// Core loaders (build-time only — called from generateStaticParams / page)
// ---------------------------------------------------------------------------

export function loadApiData(slug: string): ApiData {
  const filePath = join(API_DATA_DIR, `${slug}.json`);
  try {
    const raw = readFileSync(filePath, "utf-8");
    return JSON.parse(raw) as ApiData;
  } catch {
    throw new Error(`API data not found for slug: "${slug}". Run pnpm build:api first.`);
  }
}

export function loadApiModule(slug: string): ApiModule {
  const data = loadApiData(slug);
  if (data.kind !== "module") {
    throw new Error(`Expected module but got "${data.kind}" for slug: "${slug}"`);
  }
  return data;
}

export function loadApiClassModule(slug: string): ApiClassModule {
  const data = loadApiData(slug);
  if (data.kind !== "class-module") {
    throw new Error(`Expected class-module but got "${data.kind}" for slug: "${slug}"`);
  }
  return data;
}

// ---------------------------------------------------------------------------
// Member lookup
// ---------------------------------------------------------------------------

export function findMember(data: ApiData, memberName: string): ApiMember | undefined {
  if (data.kind === "module") {
    return data.members.find((m) => m.name === memberName);
  }
  if (data.kind === "class-module") {
    // Tensor class — methods are top-level
    const asMethod = data.methods.find((m) => m.name === memberName);
    return asMethod as ApiMember | undefined;
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// Static params helpers (used in generateStaticParams)
// ---------------------------------------------------------------------------

/** Return all valid module slugs that have a corresponding JSON file. */
export function getAllModuleSlugs(): string[] {
  const fs = require("fs") as typeof import("fs");
  try {
    return fs
      .readdirSync(API_DATA_DIR)
      .filter((f: string) => f.endsWith(".json"))
      .map((f: string) => f.replace(".json", ""));
  } catch {
    return [];
  }
}

/** Load all module metadata (name, slug, memberCount, summary) for index pages. */
export function loadModuleIndex(): Array<{
  slug: string;
  name: string;
  memberCount: number;
  summary: string | null;
}> {
  const slugs = getAllModuleSlugs();
  return slugs
    .map((slug) => {
      try {
        const data = loadApiData(slug);
        const memberCount =
          data.kind === "module"
            ? data.members.length
            : data.kind === "class-module"
              ? data.methods.length
              : 0;
        return { slug, name: data.name, memberCount, summary: data.summary ?? null };
      } catch {
        return null;
      }
    })
    .filter((x): x is NonNullable<typeof x> => x !== null);
}
