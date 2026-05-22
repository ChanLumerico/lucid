"use client";

import * as React from "react";
import Link from "next/link";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

export interface BreadcrumbSibling {
  slug: string;
  label: string;
  /** True when this entry is the *currently active* step.  The
   *  renderer highlights it inside the dropdown so users can locate
   *  themselves at a glance. */
  active?: boolean;
}

interface BreadcrumbStepProps {
  /** Display label for this step (``"nn"`` for ``lucid.nn``). */
  label: string;
  /** Navigation target for the step itself.  ``undefined`` when this
   *  is the active final segment with no deeper child (label-only). */
  href?: string;
  /** Sibling slugs at the same hierarchy level — peers of this step
   *  inside the breadcrumb path.  When non-empty, a ``▾`` caret next
   *  to the label opens a popover listing them as quick lateral jumps. */
  siblings: BreadcrumbSibling[];
}

/** One breadcrumb segment with a per-step sibling popover.  The label
 *  itself remains a regular link (clicking ``nn`` still navigates to
 *  ``/api/lucid.nn``); the caret is the affordance for "show me my
 *  siblings here so I can jump lateral".  Without this, the user's
 *  only option for lateral nav is opening the sidebar and walking the
 *  tree manually. */
export function BreadcrumbStep({ label, href, siblings }: BreadcrumbStepProps) {
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef<HTMLDivElement>(null);

  // Close on outside click — Radix Popover would handle this for us,
  // but we don't ship Popover in the bundle yet and a 30-line manual
  // implementation is the right size for this single-purpose surface.
  React.useEffect(() => {
    if (!open) return;
    function onDocClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    function onEsc(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onEsc);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onEsc);
    };
  }, [open]);

  const hasSiblings = siblings.length > 0;

  return (
    <span ref={ref} className="relative inline-flex items-center gap-0.5">
      {href ? (
        <Link
          href={href}
          className="font-mono hover:text-lucid-primary transition-colors"
        >
          {label}
        </Link>
      ) : (
        <span className="font-mono text-lucid-text-high">{label}</span>
      )}
      {hasSiblings && (
        <>
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-label={`Show ${label} siblings`}
            aria-expanded={open}
            className={cn(
              "inline-flex h-4 w-4 items-center justify-center rounded transition-colors",
              "text-lucid-text-disabled hover:text-lucid-text-mid",
              open && "text-lucid-text-mid",
            )}
          >
            <ChevronDown
              className={cn(
                "h-3 w-3 transition-transform duration-150",
                open && "rotate-180",
              )}
            />
          </button>
          {open && (
            <div
              role="menu"
              className={cn(
                "absolute left-0 top-full z-30 mt-1 w-56 origin-top-left",
                "rounded-lg border border-lucid-border bg-lucid-elevated",
                "shadow-lg shadow-black/30 py-1.5",
              )}
            >
              <p className="px-3 pb-1 text-[10px] font-semibold tracking-widest text-lucid-text-disabled uppercase">
                Siblings
              </p>
              <ul className="max-h-72 overflow-y-auto">
                {siblings.map((sib) => (
                  <li key={sib.slug}>
                    <Link
                      href={`/api/${sib.slug}`}
                      onClick={() => setOpen(false)}
                      className={cn(
                        "block px-3 py-1 font-mono text-sm transition-colors",
                        sib.active
                          ? "bg-lucid-primary/10 text-lucid-primary"
                          : "text-lucid-text-low hover:bg-lucid-surface hover:text-lucid-text-mid",
                      )}
                      role="menuitem"
                    >
                      {sib.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </span>
  );
}
