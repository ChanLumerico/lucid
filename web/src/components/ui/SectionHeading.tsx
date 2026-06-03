import type * as React from "react";
import { cn } from "@/lib/utils";

/**
 * Canonical section heading — uppercase, widely-tracked, disabled tone.  Used
 * for every member / sub-package section title.  Recipe (enforced by the
 * ``ui.section-heading`` audit contract):
 *
 *   text-xs · font-semibold · tracking-widest · uppercase · text-lucid-text-disabled
 *
 * ``className`` is for layout additions only (e.g. ``flex items-center gap-2``
 * when an anchor link sits beside the title).
 */
export function SectionHeading({
  id,
  className,
  children,
}: {
  id?: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <h2
      id={id}
      className={cn(
        "text-xs font-semibold tracking-widest uppercase text-lucid-text-disabled mb-3",
        className,
      )}
    >
      {children}
    </h2>
  );
}
