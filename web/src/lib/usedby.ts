/**
 * Loader for the ``_usedby.json`` backlink map emitted by
 * ``web/scripts/build-usedby.py``.  Each key is the canonical
 * (Griffe-resolved) path of an emitted symbol and each value is the
 * list of Lucid modules that import it.  Module-level granularity is
 * what the build script can derive cheaply from ``import`` statements
 * — finer-grained "this function calls that function" would require
 * a full call-graph analysis we deliberately skip.
 *
 * Module-level cache: the JSON is read once per Node worker and
 * reused across the request lifetime.  Safe for SSG; for SSR the
 * cache stays warm within a single deploy.
 */

import { existsSync, readFileSync } from "fs";
import { join } from "path";

interface UsedByEntry {
  module: string;
  kind: string;
}

const USEDBY_PATH = join(
  process.cwd(),
  "public",
  "api-data",
  "_usedby.json",
);

let _cache: Record<string, UsedByEntry[]> | null = null;

function _load(): Record<string, UsedByEntry[]> {
  if (_cache !== null) return _cache;
  let next: Record<string, UsedByEntry[]> = {};
  if (existsSync(USEDBY_PATH)) {
    try {
      const raw = readFileSync(USEDBY_PATH, "utf-8");
      const parsed = JSON.parse(raw);
      if (typeof parsed === "object" && parsed !== null) {
        next = parsed as Record<string, UsedByEntry[]>;
      }
    } catch {
      // Parse failure — treat as empty.
    }
  }
  _cache = next;
  return next;
}

/** Return the list of Lucid modules that import the symbol at ``path``,
 *  or an empty list when there's no recorded usage.  ``path`` should
 *  be the canonical Griffe path (``lucid.nn.module.Module``), which
 *  matches the ``path`` field on every emitted member. */
export function getUsedBy(path: string): UsedByEntry[] {
  return _load()[path] ?? [];
}
