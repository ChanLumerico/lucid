"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

interface TocEntry {
  id: string;
  text: string;
  level: number;
}

function useTocEntries(): TocEntry[] {
  const [entries, setEntries] = React.useState<TocEntry[]>([]);

  React.useEffect(() => {
    const headings = Array.from(
      document.querySelectorAll<HTMLHeadingElement>("article h2, article h3"),
    );
    // Dedup by ``id``: defensive against transient DOM states during
    // route transitions where Next may briefly mount both the outgoing
    // and incoming page's articles, and against any source of repeat
    // headings (overload constructors, etc.) so React doesn't warn
    // about duplicate keys on the ToC <li>.
    const seen = new Set<string>();
    const unique: TocEntry[] = [];
    for (const h of headings) {
      if (!h.id || seen.has(h.id)) continue;
      seen.add(h.id);
      unique.push({ id: h.id, text: h.innerText, level: Number(h.tagName[1]) });
    }
    setEntries(unique);
  }, []);

  return entries;
}

function useActiveId(ids: string[]): string {
  const [activeId, setActiveId] = React.useState("");

  React.useEffect(() => {
    if (ids.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting);
        if (visible.length > 0) {
          setActiveId(visible[0].target.id);
        }
      },
      { rootMargin: "-80px 0px -60% 0px" },
    );

    ids.forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, [ids]);

  return activeId;
}

interface PageTableOfContentsProps {
  /** Suppress rendering when the page has fewer than this many headings.
   *  Default 1 — keeps the right rail consistent across detail pages.
   *  Override to a higher value when callers want single-entry pages to
   *  hide the (admittedly redundant) ToC. */
  minEntries?: number;
}

/** Right-rail scroll-spy navigator.  Scans the page for ``<h2 id>`` /
 *  ``<h3 id>`` elements inside an ``<article>`` and renders an
 *  anchor list with the current section highlighted.  Used by both
 *  ``/docs/*`` (MDX guides) and ``/api/*`` (class detail pages with
 *  multiple methods).
 *
 *  The ToC is hidden until ``xl`` viewports — at narrower widths the
 *  rail would crowd the main content. */
export function PageTableOfContents({ minEntries = 1 }: PageTableOfContentsProps = {}) {
  const entries = useTocEntries();
  const activeId = useActiveId(entries.map((e) => e.id));

  if (entries.length < minEntries) return null;

  return (
    <aside className="hidden xl:block w-48 shrink-0">
      <div className="sticky top-24">
        <p className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-lucid-text-disabled">
          On this page
        </p>
        <nav aria-label="Table of contents">
          <ul className="space-y-1">
            {entries.map(({ id, text, level }) => (
              <li key={id}>
                <a
                  href={`#${id}`}
                  className={cn(
                    "block rounded py-0.5 text-[13px] leading-snug transition-colors duration-100",
                    level === 3 && "pl-3",
                    activeId === id
                      ? "text-lucid-primary font-medium"
                      : "text-lucid-text-low hover:text-lucid-text-mid",
                  )}
                >
                  {text}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </aside>
  );
}

/** Back-compat alias for /docs callers — generalised component sits at
 *  the same import path under its more descriptive name. */
export const DocTableOfContents = PageTableOfContents;
