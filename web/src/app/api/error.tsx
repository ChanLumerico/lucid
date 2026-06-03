"use client";

import * as React from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { buildIssueUrl } from "@/lib/error-report";

/** API-section error boundary.  Catches any render-time throw inside
 *  ``/api/*`` (broken JSON payload, missing slug, type-cast failure
 *  in a member card) and keeps the surrounding chrome — Header,
 *  Sidebar, Footer — intact.  The root ``app/error.tsx`` would
 *  re-render the entire shell which is jarring for a single-page
 *  data error.
 *
 *  Per Next.js App Router rules this MUST be a Client Component and
 *  is bounded to the ``/api`` segment by file location. */
export default function ApiError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  React.useEffect(() => {
    if (process.env.NODE_ENV !== "production") {
      // eslint-disable-next-line no-console
      console.error("[lucid-docs] /api page error", error);
    }
  }, [error]);

  const issueUrl = buildIssueUrl({
    title: `[docs] /api error: ${error.message}`,
    error,
    section: "/api",
  });

  return (
    <div className="mx-auto max-w-2xl px-4 py-16 text-center">
      <p className="mb-3 font-mono text-[11px] font-semibold tracking-widest uppercase text-lucid-error">
        page error
      </p>
      <h1 className="font-mono text-2xl sm:text-3xl font-bold text-lucid-text-high mb-3">
        Couldn&rsquo;t render this API page
      </h1>
      <p className="text-sm text-lucid-text-mid leading-relaxed mb-2 max-w-prose mx-auto">
        Most likely the underlying JSON payload is stale or malformed.
        Try reloading — the prebuild step regenerates the API JSONs from
        Lucid&rsquo;s Python sources on every <code className="font-mono text-lucid-text-low">pnpm dev</code> /{" "}
        <code className="font-mono text-lucid-text-low">pnpm build</code> kickoff.
      </p>
      {error.digest && (
        <p className="font-mono text-[10px] text-lucid-text-disabled mb-6">
          digest <span className="text-lucid-text-low">{error.digest}</span>
        </p>
      )}
      <div className="flex flex-wrap items-center justify-center gap-3 mt-6">
        <Button onClick={reset}>Try again</Button>
        <Button variant="secondary" asChild>
          <Link href="/api">API home</Link>
        </Button>
        <Button variant="ghost" asChild>
          <a href={issueUrl} target="_blank" rel="noopener noreferrer">
            Report this
          </a>
        </Button>
      </div>
    </div>
  );
}
