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
    setEntries(
      headings
        .filter((h) => h.id)
        .map((h) => ({
          id: h.id,
          text: h.innerText,
          level: Number(h.tagName[1]),
        })),
    );
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

export function DocTableOfContents() {
  const entries = useTocEntries();
  const activeId = useActiveId(entries.map((e) => e.id));

  if (entries.length === 0) return null;

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
