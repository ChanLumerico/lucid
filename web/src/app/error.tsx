"use client";

import * as React from "react";
import { Button } from "@/components/ui/button";

/** Route-level error boundary.  Next.js mounts this when a server
 *  component (or any descendant) throws during render.  Resets via the
 *  ``reset()`` callback Next provides, which retries the render. */
export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  React.useEffect(() => {
    // Surface the digest in the dev console — the production stack is
    // stripped at build time.
    if (process.env.NODE_ENV !== "production") {
      // eslint-disable-next-line no-console
      console.error("[lucid-docs] page render error", error);
    }
  }, [error]);

  return (
    <div className="min-h-dvh flex items-center justify-center px-4">
      <div className="mx-auto max-w-md text-center">
        <p className="mb-3 font-mono text-[11px] font-semibold tracking-widest uppercase text-lucid-error">
          render error
        </p>
        <h1 className="font-mono text-3xl sm:text-4xl font-bold text-lucid-text-high mb-3">
          Something broke while rendering this page
        </h1>
        <p className="text-sm text-lucid-text-mid leading-relaxed mb-2">
          The docs site hit an unexpected error.  Try reloading — the build
          pipeline regenerates JSON from source on every dev save, so most
          render-time issues self-heal on the next pass.
        </p>
        {error.digest && (
          <p className="font-mono text-[10px] text-lucid-text-disabled mb-6">
            digest <span className="text-lucid-text-low">{error.digest}</span>
          </p>
        )}
        <div className="flex flex-wrap items-center justify-center gap-3">
          <Button onClick={reset}>Try again</Button>
          <Button variant="secondary" asChild>
            <a href="/api">Back to API Reference</a>
          </Button>
        </div>
      </div>
    </div>
  );
}
