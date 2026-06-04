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

/** Per-section scroll progress in [0, 1].  Reads each section's
 *  document position on every scroll tick and computes "how far past
 *  the section's top has the viewport scrolled, normalised against
 *  the section's own height plus any trailing whitespace before the
 *  next section".  ``rAF``-throttled so even on a long page we run
 *  the calc at most once per frame.
 *
 *  Returns a stable ``Map`` reference per render — consumers should
 *  read values with ``progress.get(id) ?? 0``. */
function useTocProgress(ids: string[]): Map<string, number> {
  const [progress, setProgress] = React.useState<Map<string, number>>(
    () => new Map(),
  );

  React.useEffect(() => {
    if (ids.length === 0) return;
    let rafId = 0;

    function compute() {
      const next = new Map<string, number>();
      const viewportTop = window.scrollY;
      const viewportHeight = window.innerHeight;
      const viewportMid = viewportTop + viewportHeight * 0.3;

      // Collect each section's absolute top so we can determine its
      // end-of-section position from the *next* section's top —
      // headings don't carry a height that maps to "section size", we
      // have to derive it from gap-to-next.
      const sections: Array<{ id: string; top: number }> = [];
      for (const id of ids) {
        const el = document.getElementById(id);
        if (!el) continue;
        sections.push({
          id,
          top: el.getBoundingClientRect().top + window.scrollY,
        });
      }
      sections.sort((a, b) => a.top - b.top);

      for (let i = 0; i < sections.length; i++) {
        const { id, top } = sections[i];
        const nextTop =
          i + 1 < sections.length
            ? sections[i + 1].top
            : document.documentElement.scrollHeight;
        const sectionHeight = Math.max(1, nextTop - top);
        const raw = (viewportMid - top) / sectionHeight;
        next.set(id, Math.min(1, Math.max(0, raw)));
      }
      setProgress(next);
    }

    function onScroll() {
      if (rafId) return;
      rafId = requestAnimationFrame(() => {
        rafId = 0;
        compute();
      });
    }

    compute();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll, { passive: true });
    return () => {
      if (rafId) cancelAnimationFrame(rafId);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, [ids]);

  return progress;
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
  const ids = React.useMemo(() => entries.map((e) => e.id), [entries]);
  const activeId = useActiveId(ids);
  const progress = useTocProgress(ids);

  if (entries.length < minEntries) return null;

  // Smooth in-page navigation.  ``scrollIntoView`` honours the heading's
  // ``scroll-mt-24`` so the target clears the fixed header; ``replaceState``
  // syncs the URL hash without piling history entries on every TOC click.
  const onEntryClick = (e: React.MouseEvent, id: string) => {
    const el = document.getElementById(id);
    if (!el) return;
    e.preventDefault();
    el.scrollIntoView({ behavior: "smooth", block: "start" });
    history.replaceState(null, "", `#${id}`);
  };

  return (
    // The ASIDE itself is sticky: its containing block is the tall flex row, so
    // it follows the scroll.  (A sticky *child* fails here because the parent
    // ``items-start`` shrinks this aside to its own height — no room to stick.)
    // ``max-h`` + ``overflow-y-auto`` lets long ToCs (Tensor has ~88 entries)
    // scroll internally instead of overflowing the viewport.
    <aside className="hidden xl:block w-48 shrink-0 self-start sticky top-24 max-h-[calc(100dvh-7rem)] overflow-y-auto overscroll-contain pb-6">
      <div>
        <p className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-lucid-text-disabled">
          On this page
        </p>
        <nav aria-label="Table of contents">
          {/* Each entry gets a 2 px-wide track on the left; the
              ``span`` inside fills bottom-up from 0 → 100 % matching
              the per-section scroll progress.  At rest the track is
              the same colour as the panel border so empty progress
              reads as a clean rail.  Once you scroll past a section,
              its bar stays full so users see how far they've worked
              through the page at a glance. */}
          <ul className="space-y-1 border-l border-lucid-border/60">
            {entries.map(({ id, text, level }) => {
              const p = Math.round((progress.get(id) ?? 0) * 100);
              return (
                <li key={id} className="relative">
                  <span
                    aria-hidden
                    className="pointer-events-none absolute -left-px top-0 w-px bg-lucid-primary/70 transition-[height] duration-150"
                    style={{ height: `${p}%` }}
                  />
                  <a
                    href={`#${id}`}
                    onClick={(e) => onEntryClick(e, id)}
                    className={cn(
                      "block rounded py-0.5 pl-2 text-[13px] leading-snug transition-colors duration-100",
                      level === 3 && "pl-5",
                      activeId === id
                        ? "text-lucid-primary font-medium"
                        : "text-lucid-text-low hover:text-lucid-text-mid",
                    )}
                  >
                    {text}
                  </a>
                </li>
              );
            })}
          </ul>
        </nav>
      </div>
    </aside>
  );
}

/** Back-compat alias for /docs callers — generalised component sits at
 *  the same import path under its more descriptive name. */
export const DocTableOfContents = PageTableOfContents;
