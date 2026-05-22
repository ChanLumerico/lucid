"use client";

import * as React from "react";
import { Link2, Check } from "lucide-react";
import { cn } from "@/lib/utils";

interface AnchorLinkProps {
  /** Anchor id of the heading this control belongs to.  The button
   *  copies ``<origin>/<pathname>#<id>`` to the clipboard. */
  id: string;
  className?: string;
}

/** Hover-reveal copy-anchor control rendered next to ``<h2>`` / ``<h3>``
 *  headings.  Standard docs UX pattern (GitHub, MDN, …) — click the
 *  ``#`` icon and the deep-link to that section is on the clipboard
 *  ready to paste into a chat / issue / commit message.
 *
 *  The button is invisible until the parent element (the heading or
 *  its wrapping section) gets ``:hover`` or keyboard focus — at rest
 *  it doesn't visually compete with the heading text. */
export function AnchorLink({ id, className }: AnchorLinkProps) {
  const [copied, setCopied] = React.useState(false);

  const handleClick = React.useCallback(
    async (e: React.MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
      const url =
        typeof window !== "undefined"
          ? `${window.location.origin}${window.location.pathname}#${id}`
          : `#${id}`;
      try {
        await navigator.clipboard.writeText(url);
        // Also update the address bar so the back button works as
        // expected after copy — small but appreciated.
        if (typeof window !== "undefined") {
          window.history.replaceState(null, "", `#${id}`);
        }
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      } catch {
        // Clipboard API blocked (insecure context) — fall back to
        // just updating the URL.  User can still copy from address bar.
        if (typeof window !== "undefined") {
          window.history.replaceState(null, "", `#${id}`);
        }
      }
    },
    [id],
  );

  // Visibility: ``hidden`` (display: none) instead of ``opacity-0`` so
  // the icon vacates the flex slot entirely when the parent isn't
  // hovered.  Earlier ``opacity-0`` version still reserved its 20 px
  // box + 8 px flex gap on each side, which made the trailing return
  // type look orphaned from the function name at rest.  Toggling
  // ``display`` causes a tiny one-frame width shift on hover, but the
  // icon is small enough that it reads as a reveal, not a jump.
  return (
    <button
      type="button"
      onClick={handleClick}
      aria-label={copied ? "Link copied" : "Copy link to section"}
      className={cn(
        "h-5 w-5 items-center justify-center rounded transition-colors duration-150",
        copied
          ? "inline-flex text-lucid-success"
          : cn(
              "hidden group-hover:inline-flex group-focus-within:inline-flex",
              "text-lucid-text-disabled hover:text-lucid-primary",
            ),
        className,
      )}
    >
      {copied ? <Check className="h-3 w-3" /> : <Link2 className="h-3 w-3" />}
    </button>
  );
}
