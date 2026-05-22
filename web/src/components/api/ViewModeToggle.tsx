"use client";

import * as React from "react";
import { Rows3, AlignJustify } from "lucide-react";
import { cn } from "@/lib/utils";

const STORAGE_KEY = "lucid-docs-module-view";

type ViewMode = "detailed" | "compact";

/** Module-overview view-mode toggle.  ``detailed`` (default) renders
 *  every member as a 2-line card with name + return type + summary
 *  excerpt.  ``compact`` collapses each card to a single line — just
 *  badge + name + return type — for fast scanning on big modules
 *  (``lucid.nn.functional`` has 121 members).
 *
 *  The mode is persisted in ``localStorage`` so a single click sticks
 *  across navigations.  Applied via ``data-view`` on the
 *  ``<article>`` ancestor; Tailwind ``[data-view=...]:`` variants in
 *  the card components control the actual layout switch — no double
 *  render.  Until the effect runs the page renders ``detailed`` (safe
 *  default + matches SSR output, so hydration mismatch is impossible). */
export function ViewModeToggle() {
  const [mode, setMode] = React.useState<ViewMode>("detailed");
  const [mounted, setMounted] = React.useState(false);

  React.useEffect(() => {
    setMounted(true);
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored === "compact" || stored === "detailed") {
        setMode(stored);
        _applyMode(stored);
      }
    } catch {
      // localStorage blocked — accept the default.
    }
  }, []);

  const setAndPersist = React.useCallback((next: ViewMode) => {
    setMode(next);
    _applyMode(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // Storage disabled — non-fatal.
    }
  }, []);

  return (
    <div
      className={cn(
        "inline-flex rounded-lg border border-lucid-border bg-lucid-surface p-0.5",
        // Pre-mount: render in a neutral state so the button doesn't
        // flash an "active" highlight that disagrees with the stored
        // preference for half a frame after hydration.
        !mounted && "opacity-90",
      )}
      role="group"
      aria-label="Module view density"
    >
      <button
        type="button"
        onClick={() => setAndPersist("detailed")}
        aria-pressed={mode === "detailed"}
        aria-label="Detailed view — show summaries"
        className={cn(
          "flex h-7 w-7 items-center justify-center rounded transition-colors",
          mode === "detailed"
            ? "bg-lucid-elevated text-lucid-text-high"
            : "text-lucid-text-low hover:text-lucid-text-mid",
        )}
      >
        <Rows3 className="h-3.5 w-3.5" />
      </button>
      <button
        type="button"
        onClick={() => setAndPersist("compact")}
        aria-pressed={mode === "compact"}
        aria-label="Compact view — signatures only"
        className={cn(
          "flex h-7 w-7 items-center justify-center rounded transition-colors",
          mode === "compact"
            ? "bg-lucid-elevated text-lucid-text-high"
            : "text-lucid-text-low hover:text-lucid-text-mid",
        )}
      >
        <AlignJustify className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}

/** Walk up from the toggle to find the nearest module-overview
 *  ``<article>`` and stamp ``data-view`` on it.  We intentionally do
 *  not target ``document.documentElement`` — the preference is per-page
 *  scoped (a user might want compact for ``nn.functional`` but detailed
 *  elsewhere) and document-level state would leak across navigations. */
function _applyMode(mode: ViewMode): void {
  if (typeof document === "undefined") return;
  const article = document.querySelector<HTMLElement>("article[data-module-overview]");
  if (article) article.dataset.view = mode;
}
