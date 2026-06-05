import type * as React from "react";
import { cn } from "@/lib/utils";

/**
 * Canonical subsection heading (``h4``) — the same uppercase / widely-tracked
 * / disabled-tone recipe as {@link SectionHeading}, but for the smaller
 * in-card subsections (Parameters, Returns, Examples, Model Size, Attributes,
 * See also, Used by).  These cards repeated the recipe inline ~10× and the
 * ``ui.section-heading`` audit contract only covers the ``h2`` section titles,
 * so the ``h4`` copies could silently drift — this is their single source.
 *
 * ``className`` is for layout additions only (e.g. ``flex items-center gap-2``).
 */
export function SubsectionHeading({
  className,
  children,
}: {
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <h4
      className={cn(
        "text-xs font-semibold tracking-widest text-lucid-text-disabled uppercase",
        className,
      )}
    >
      {children}
    </h4>
  );
}
