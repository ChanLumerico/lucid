"use client";

import * as React from "react";
import { usePathname } from "next/navigation";
import { addRecentPage } from "@/lib/recent-pages";

/** Layout-level tracker that records every distinct ``/api/*`` or
 *  ``/docs/*`` page the user visits.  Reads ``document.title`` on a
 *  short ``rAF`` delay so Next.js has time to update the title for
 *  the freshly-mounted route before we snapshot it.
 *
 *  Mounted once at the docs root so any pathway into a page —
 *  sidebar click, search dialog navigate, direct URL, back/forward —
 *  funnels through this single record point. */
export function RecentPageTracker() {
  const pathname = usePathname();

  React.useEffect(() => {
    // Skip the landing page / category indexes — those don't carry
    // useful per-symbol context for the recents list.
    if (!pathname) return;
    const tracked =
      pathname.startsWith("/api/") || pathname.startsWith("/docs/");
    if (!tracked) return;

    // ``rAF`` gives ``<Metadata>`` one frame to commit the new title;
    // a microtask is too early and we'd capture the previous page's
    // title.  Bail if the component unmounts before the frame fires.
    let cancelled = false;
    const id = requestAnimationFrame(() => {
      if (cancelled) return;
      const title = document.title || pathname;
      // Strip the trailing ``" — Lucid"`` / ``" | Lucid"`` site
      // suffix so the recents list reads as just the page name.
      const cleanTitle = title.replace(/\s*[|—–]\s*Lucid.*$/u, "").trim();
      addRecentPage({ href: pathname, title: cleanTitle || pathname });
    });
    return () => {
      cancelled = true;
      cancelAnimationFrame(id);
    };
  }, [pathname]);

  return null;
}
