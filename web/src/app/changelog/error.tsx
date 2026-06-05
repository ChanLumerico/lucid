"use client";

import * as React from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { buildIssueUrl } from "@/lib/error-report";

/** Changelog-section error boundary.  Mirrors ``app/api/error.tsx`` and
 *  ``app/docs/error.tsx`` so a client-side render throw on ``/changelog``
 *  (e.g. a malformed changelog entry) is caught in-place and keeps the
 *  surrounding chrome intact, instead of bubbling to the root boundary.
 *
 *  Must be a Client Component (Next.js App Router rule) and is scoped to
 *  the ``/changelog`` segment by file location. */
export default function ChangelogError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  React.useEffect(() => {
    if (process.env.NODE_ENV !== "production") {
      // eslint-disable-next-line no-console
      console.error("[lucid-docs] /changelog page error", error);
    }
  }, [error]);

  const issueUrl = buildIssueUrl({
    title: `[docs] /changelog error: ${error.message}`,
    error,
    section: "/changelog",
  });

  return (
    <div className="mx-auto max-w-2xl px-4 py-16 text-center">
      <p className="mb-3 font-mono text-[11px] font-semibold tracking-widest uppercase text-lucid-error">
        page error
      </p>
      <h1 className="font-mono text-2xl sm:text-3xl font-bold text-lucid-text-high mb-3">
        Couldn&rsquo;t render the changelog
      </h1>
      <p className="text-sm text-lucid-text-mid leading-relaxed mb-2 max-w-prose mx-auto">
        The changelog failed to render. Try reloading — the page is rebuilt
        from <code className="font-mono text-lucid-text-low">CHANGELOG.md</code>{" "}
        on every build.
      </p>
      {error.digest && (
        <p className="font-mono text-[10px] text-lucid-text-disabled mb-6">
          digest <span className="text-lucid-text-low">{error.digest}</span>
        </p>
      )}
      <div className="flex flex-wrap items-center justify-center gap-3 mt-6">
        <Button onClick={reset}>Try again</Button>
        <Button variant="secondary" asChild>
          <Link href="/changelog">Changelog home</Link>
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
