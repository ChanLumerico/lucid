"use client";

import * as React from "react";
import { usePathname } from "next/navigation";

/** Top-edge progress indicator for route transitions.  Next.js App
 *  Router doesn't expose a navigation-start event (only the resolved
 *  pathname after the new render commits), so we approximate the bar
 *  with an interception model:
 *
 *    1. Click on any in-app ``<a href="/…">`` flips us into ``loading``,
 *       starting the bar from 0 → 70 % over ~400 ms (slow ramp so
 *       the bar isn't already full when fast navigations complete).
 *    2. ``usePathname`` change ends the loading state — bar
 *       completes to 100 %, fades, resets.
 *    3. Safety net: 4 s timeout to clear the loading state even when
 *       navigation never settles (offline, error boundary triggered).
 *
 *  Mounted once at the layout root.  No fetch, no library, ~80 lines.
 */
export function RouteProgress() {
  const pathname = usePathname();
  const [active, setActive] = React.useState(false);
  const [percent, setPercent] = React.useState(0);
  const lastPathRef = React.useRef(pathname);
  const rampTimerRef = React.useRef<ReturnType<typeof setInterval> | null>(null);
  const fadeTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const watchdogRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  function _clearTimers() {
    if (rampTimerRef.current) {
      clearInterval(rampTimerRef.current);
      rampTimerRef.current = null;
    }
    if (fadeTimerRef.current) {
      clearTimeout(fadeTimerRef.current);
      fadeTimerRef.current = null;
    }
    if (watchdogRef.current) {
      clearTimeout(watchdogRef.current);
      watchdogRef.current = null;
    }
  }

  // Listen for in-app link clicks.  We intentionally use bubble-phase
  // ``click`` so the click is already past any ``preventDefault`` from
  // the application — if the click was cancelled, our progress bar
  // would have shown for nothing.
  React.useEffect(() => {
    function onClick(e: MouseEvent) {
      if (e.defaultPrevented) return;
      if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
      const target = e.target;
      if (!(target instanceof Element)) return;
      const anchor = target.closest("a");
      if (!anchor) return;
      const href = anchor.getAttribute("href");
      if (!href) return;
      // Only fire on internal navigation — external links open in a
      // new tab anyway, and the visible-progress wouldn't survive the
      // page unload.
      if (
        href.startsWith("http")
        || href.startsWith("//")
        || anchor.target === "_blank"
      ) {
        return;
      }
      // Hash-only links are intra-page and don't trigger a route change.
      if (href.startsWith("#")) return;

      _clearTimers();
      setActive(true);
      setPercent(8);
      // Slow ramp toward 70 % so even the slowest Next render still
      // looks like it's making progress; 70 % is the canonical
      // nprogress-style ceiling that leaves room for the final-jump
      // animation on completion.
      let p = 8;
      rampTimerRef.current = setInterval(() => {
        p = Math.min(70, p + (70 - p) * 0.12);
        setPercent(p);
      }, 80);
      // Hard cap: never show the bar for more than 4 s — at that
      // point either nav failed silently or the page is huge enough
      // that the user already gave up on the indicator.
      watchdogRef.current = setTimeout(() => {
        _clearTimers();
        setActive(false);
        setPercent(0);
      }, 4000);
    }
    document.addEventListener("click", onClick);
    return () => {
      document.removeEventListener("click", onClick);
      _clearTimers();
    };
  }, []);

  // Pathname change = render committed.  Snap to 100 %, fade out.
  React.useEffect(() => {
    if (pathname === lastPathRef.current) return;
    lastPathRef.current = pathname;
    if (!active) return;
    if (rampTimerRef.current) {
      clearInterval(rampTimerRef.current);
      rampTimerRef.current = null;
    }
    setPercent(100);
    fadeTimerRef.current = setTimeout(() => {
      setActive(false);
      setPercent(0);
    }, 250);
    return () => {
      if (fadeTimerRef.current) clearTimeout(fadeTimerRef.current);
    };
  }, [pathname, active]);

  if (!active && percent === 0) return null;

  return (
    <div
      aria-hidden
      className="fixed inset-x-0 top-0 z-[60] h-0.5 pointer-events-none"
    >
      <div
        className="h-full bg-lucid-primary transition-[width,opacity] duration-150"
        style={{
          width: `${percent}%`,
          opacity: active ? 1 : 0,
          boxShadow: "0 0 8px var(--color-lucid-primary)",
        }}
      />
    </div>
  );
}
