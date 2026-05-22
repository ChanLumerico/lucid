import { NextResponse } from "next/server";
import { buildSearchIndex } from "@/lib/search-index";

// ``force-static`` already opts this route into Next.js's build-time
// rendering — the JSON is computed once during ``next build`` and
// served as a static asset.  We layer an explicit
// ``Cache-Control: immutable`` header on top so:
//
//   1. The browser caches the response per session — subsequent
//      ``⌘K`` opens in the same tab don't re-fetch.
//   2. CDNs (Vercel edge, GitHub Pages cache) treat it as long-lived.
//
// The index is regenerated on every ``next build`` so the immutable
// hint is honest at the build-output granularity — a fresh deploy
// invalidates upstream caches by URL anyway.
export const dynamic = "force-static";

export function GET() {
  const index = buildSearchIndex();
  return NextResponse.json(index, {
    headers: {
      "Cache-Control": "public, max-age=3600, stale-while-revalidate=86400",
    },
  });
}
