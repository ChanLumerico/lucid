import Link from "next/link";
import type { SeeAlsoItem } from "@/lib/types";
import { getAllModuleSlugs } from "@/lib/api-loader";
import { MathText } from "./MathText";

interface SeeAlsoBlockProps {
  items: SeeAlsoItem[];
}

/** Resolve a docstring ``See Also`` entry name to a docs-site URL.
 *
 *  Resolution policy (mirrors ``tools/check_docstring_xrefs.py``):
 *    1. Fully-qualified path that names a module slug directly
 *       (``lucid.nn.functional``) → ``/api/<slug>``.
 *    2. Dotted path where the prefix is a module slug and the leaf is
 *       a member of that module (``lucid.nn.Linear`` →
 *       ``/api/lucid.nn/Linear``).
 *    3. Bare name where exactly one emitted module exposes it
 *       (``dropout`` → ``/api/lucid.nn.functional/dropout``) — best-effort.
 *    4. Otherwise: return ``null`` and the renderer falls back to
 *       plain text. */
function resolveLink(name: string, slugs: Set<string>): string | null {
  if (slugs.has(name)) return `/api/${name}`;
  // Try splitting from the right: longest possible module slug + leaf.
  const parts = name.split(".");
  for (let i = parts.length - 1; i > 0; i--) {
    const candidate = parts.slice(0, i).join(".");
    if (slugs.has(candidate)) {
      const leaf = parts.slice(i).join(".");
      return `/api/${candidate}/${leaf}`;
    }
  }
  return null;
}

export function SeeAlsoBlock({ items }: SeeAlsoBlockProps) {
  if (!items || items.length === 0) return null;
  const slugs = new Set(getAllModuleSlugs());
  return (
    <section className="space-y-2">
      <h4 className="text-xs font-semibold tracking-widest text-lucid-text-disabled uppercase">
        See Also
      </h4>
      <ul className="rounded-xl border-l-2 border-lucid-primary/40 bg-lucid-primary/5 px-4 py-3 space-y-1.5">
        {items.map((item, i) => {
          const href = resolveLink(item.name, slugs);
          const nameEl = href ? (
            <Link
              href={href}
              className="font-mono text-sm text-lucid-primary hover:text-lucid-primary-light transition-colors"
            >
              {item.name}
            </Link>
          ) : (
            <code className="font-mono text-sm text-lucid-text-mid">{item.name}</code>
          );
          return (
            <li
              key={i}
              className="flex flex-col gap-0.5 sm:flex-row sm:items-baseline sm:gap-2"
            >
              <span className="shrink-0">{nameEl}</span>
              {item.description && (
                <span className="text-sm text-lucid-text-mid">
                  <span className="text-lucid-text-disabled mx-1">—</span>
                  <MathText text={item.description} className="inline" />
                </span>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
