"use client";

import * as React from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { buildIssueUrl } from "@/lib/error-report";

/** Docs-section error boundary.  Same role as ``app/api/error.tsx`` —
 *  keeps Header / Sidebar / Footer mounted and only re-renders the
 *  main column when an MDX guide throws on render.  Most likely cause
 *  is malformed frontmatter or a KaTeX strict-mode rejection inside a
 *  guide; both heal on the next file save. */
export default function DocsError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  React.useEffect(() => {
    if (process.env.NODE_ENV !== "production") {
      // eslint-disable-next-line no-console
      console.error("[lucid-docs] /docs page error", error);
    }
  }, [error]);

  const issueUrl = buildIssueUrl({
    title: `[docs] /docs error: ${error.message}`,
    error,
    section: "/docs",
  });

  return (
    <div className="mx-auto max-w-2xl px-4 py-16 text-center">
      <p className="mb-3 font-mono text-[11px] font-semibold tracking-widest uppercase text-lucid-error">
        page error
      </p>
      <h1 className="font-mono text-2xl sm:text-3xl font-bold text-lucid-text-high mb-3">
        Couldn&rsquo;t render this guide
      </h1>
      <p className="text-sm text-lucid-text-mid leading-relaxed mb-2 max-w-prose mx-auto">
        MDX or KaTeX likely choked on the source.  Try reloading; if it
        recurs, check the file&rsquo;s frontmatter and inline math.
      </p>
      {error.digest && (
        <p className="font-mono text-[10px] text-lucid-text-disabled mb-6">
          digest <span className="text-lucid-text-low">{error.digest}</span>
        </p>
      )}
      <div className="flex flex-wrap items-center justify-center gap-3 mt-6">
        <Button onClick={reset}>Try again</Button>
        <Button variant="secondary" asChild>
          <Link href="/docs">Docs home</Link>
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
