"use client";

import * as React from "react";
import { ArrowUp } from "lucide-react";
import { cn } from "@/lib/utils";

/** Floating "back to top" affordance.  Hidden until the reader has
 *  scrolled past ~400 px (where the page-header anchor stops being
 *  visible), then fades in at the bottom-right corner.  Click /
 *  Enter / Space all return to the top with smooth scrolling. */
export function BackToTop({ threshold = 400 }: { threshold?: number }) {
  const [visible, setVisible] = React.useState(false);

  React.useEffect(() => {
    const onScroll = () => setVisible(window.scrollY > threshold);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [threshold]);

  const onClick = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <button
      type="button"
      onClick={onClick}
      aria-label="Back to top"
      className={cn(
        "fixed bottom-6 right-6 z-40",
        "inline-flex h-10 w-10 items-center justify-center rounded-full",
        "border border-lucid-border bg-lucid-surface/90 backdrop-blur",
        "text-lucid-text-mid hover:text-lucid-primary hover:border-lucid-primary/50",
        "transition-all duration-200",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-lucid-primary/60",
        visible ? "opacity-100 translate-y-0 pointer-events-auto" : "opacity-0 translate-y-2 pointer-events-none",
      )}
    >
      <ArrowUp className="h-4 w-4" aria-hidden />
    </button>
  );
}
