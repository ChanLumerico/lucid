"use client";

import * as React from "react";

/** Thin progress bar at the top of the viewport indicating how far the
 *  reader has scrolled through the current page.  Renders as a
 *  fixed-position 2px slab below the sticky header (top-14 = 56 px
 *  matches the header's height).  Updated via ``requestAnimationFrame``
 *  so the work stays off the scroll-listener critical path.
 *
 *  Hidden when the document is shorter than the viewport — a bar
 *  pinned at 0% on a non-scrollable page is just noise. */
export function ScrollProgress() {
  const [pct, setPct] = React.useState(0);
  const [active, setActive] = React.useState(false);

  React.useEffect(() => {
    let raf = 0;
    const tick = () => {
      const doc = document.documentElement;
      const scrolled = window.scrollY;
      const total = doc.scrollHeight - window.innerHeight;
      if (total <= 0) {
        setActive(false);
        return;
      }
      setActive(true);
      const next = Math.min(100, Math.max(0, (scrolled / total) * 100));
      setPct(next);
    };
    const onScroll = () => {
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = 0;
        tick();
      });
    };
    tick();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  if (!active) return null;
  return (
    <div
      role="progressbar"
      aria-label="Page scroll progress"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(pct)}
      className="fixed inset-x-0 top-14 z-40 h-0.5 pointer-events-none"
    >
      <div
        className="h-full bg-gradient-to-r from-lucid-primary via-lucid-primary-light to-lucid-blue transition-[width] duration-100 ease-out"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}
