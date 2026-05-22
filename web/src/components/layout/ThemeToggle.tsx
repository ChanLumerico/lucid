"use client";

import * as React from "react";
import { Sun, Moon } from "lucide-react";
import { useTheme } from "./ThemeProvider";
import { cn } from "@/lib/utils";

/** Header-mounted sun/moon button.  Toggles the html ``data-theme``
 *  attribute via :func:`useTheme` — the CSS layer in ``globals.css``
 *  swaps every ``--color-lucid-*`` variable accordingly. */
export function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const [mounted, setMounted] = React.useState(false);

  // Wait one tick after hydration before showing the toggle — until
  // then the ThemeProvider hasn't synced its state with the DOM yet,
  // and rendering the wrong icon would cause a visible flip on the
  // first interactive frame.
  React.useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
      className={cn(
        "inline-flex h-8 w-8 items-center justify-center rounded-md",
        "text-lucid-text-low hover:text-lucid-text-mid hover:bg-lucid-elevated",
        "transition-colors",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-lucid-primary/60",
      )}
    >
      {mounted ? (
        theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />
      ) : (
        // Placeholder same-size icon during the pre-hydration tick so
        // the header layout doesn't shift when the real icon resolves.
        <Sun className="h-4 w-4 opacity-0" aria-hidden />
      )}
    </button>
  );
}
