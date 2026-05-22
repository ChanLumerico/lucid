// Build-time metadata loader.  Emitted by `scripts/build-meta.py` at
// prebuild and read here at module-import time.  Stays server-side: the
// JSON sits in `public/` so Next.js can also serve it as a raw URL, but
// component code should `import { buildMeta } from "@/lib/build-meta"`
// to avoid an extra HTTP round-trip.

import { readFileSync } from "node:fs";
import { join } from "node:path";

export interface BuildMeta {
  lucid_version: string | null;
  git_sha: string | null;
  git_sha_full: string | null;
  git_branch: string | null;
  built_at: string | null;
}

const META_PATH = join(process.cwd(), "public", "build-meta.json");

const FALLBACK: BuildMeta = {
  lucid_version: null,
  git_sha: null,
  git_sha_full: null,
  git_branch: null,
  built_at: null,
};

function loadMeta(): BuildMeta {
  try {
    const raw = readFileSync(META_PATH, "utf-8");
    return { ...FALLBACK, ...(JSON.parse(raw) as Partial<BuildMeta>) };
  } catch {
    // build-meta.json missing — happens during fresh checkouts before
    // `pnpm prebuild` has run.  Returning the fallback keeps page
    // rendering working; the UI shows "—" placeholders.
    return FALLBACK;
  }
}

export const buildMeta: BuildMeta = loadMeta();

/** Construct a GitHub source URL anchored to the SHA recorded at build
 *  time, so a "View source" link points at the exact tree the docs were
 *  generated from rather than drifting against ``main``. */
export function sourceUrl(relPath: string, line?: number): string | null {
  const sha = buildMeta.git_sha_full ?? buildMeta.git_branch;
  if (!sha) return null;
  const lineFragment = line ? `#L${line}` : "";
  return `https://github.com/ChanLumerico/lucid/blob/${sha}/${relPath}${lineFragment}`;
}
