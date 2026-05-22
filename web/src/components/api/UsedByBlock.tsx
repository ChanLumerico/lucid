import Link from "next/link";
import { getUsedBy } from "@/lib/usedby";
import { getAllModuleSlugs } from "@/lib/api-loader";
import { cn } from "@/lib/utils";

interface UsedByBlockProps {
  /** Canonical Griffe path of the symbol — exactly what ``path`` is on
   *  the ``ApiFunction`` / ``ApiClass`` payload.  We never accept the
   *  slug form because re-export paths don't appear as keys in
   *  ``_usedby.json``. */
  path: string;
  /** Cap rendered entries.  When the symbol is widely used (Module,
   *  Tensor, …) we'd otherwise produce a 50-row wall; this limit
   *  keeps the block scannable and an explicit footer says how many
   *  were hidden. */
  maxRows?: number;
}

/** Where can I go from here? section.  Lists every Lucid module that
 *  imports the current symbol, one per row, linked to that module's
 *  docs page when one exists.  Renders nothing when the backlink map
 *  has no entries for the symbol — the docs site shouldn't grow an
 *  empty "Used by" placeholder on every leaf page. */
export function UsedByBlock({ path, maxRows = 12 }: UsedByBlockProps) {
  const rows = getUsedBy(path);
  if (rows.length === 0) return null;

  // Only link to modules that have an emitted JSON — we own that set
  // already via ``getAllModuleSlugs``.  Non-emitted importers
  // (private leaf modules, ``lucid._dispatch``, …) still appear as
  // plain text so the count stays honest.
  const slugs = new Set(getAllModuleSlugs());
  const visible = rows.slice(0, maxRows);
  const hiddenCount = rows.length - visible.length;

  return (
    <section className="space-y-2">
      <h4 className="text-xs font-semibold tracking-widest text-lucid-text-disabled uppercase">
        Used by{" "}
        <span className="ml-1 font-mono text-[10px] text-lucid-text-low">
          {rows.length}
        </span>
      </h4>
      <ul
        className={cn(
          "rounded-xl border border-lucid-border bg-lucid-surface/40 px-4 py-3",
          "grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-1",
        )}
      >
        {visible.map((row) => {
          const moduleSlug = _resolveSlug(row.module, slugs);
          return (
            <li key={row.module} className="min-w-0">
              {moduleSlug ? (
                <Link
                  href={`/api/${moduleSlug}`}
                  className="font-mono text-[12px] text-lucid-primary/80 hover:text-lucid-primary truncate block"
                >
                  {row.module}
                </Link>
              ) : (
                <span className="font-mono text-[12px] text-lucid-text-low truncate block">
                  {row.module}
                </span>
              )}
            </li>
          );
        })}
      </ul>
      {hiddenCount > 0 && (
        <p className="text-[11px] text-lucid-text-disabled">
          … {hiddenCount} more
        </p>
      )}
    </section>
  );
}

/** Resolve a Python module path to the docs slug if one exists.  When
 *  no exact slug matches, try walking parent prefixes — ``lucid.nn.modules.conv``
 *  is documented as the page ``lucid.nn`` with ``conv`` as a sub-grouping. */
function _resolveSlug(modulePath: string, slugs: Set<string>): string | null {
  if (slugs.has(modulePath)) return modulePath;
  const parts = modulePath.split(".");
  for (let i = parts.length - 1; i > 0; i--) {
    const prefix = parts.slice(0, i).join(".");
    if (slugs.has(prefix)) return prefix;
  }
  return null;
}
