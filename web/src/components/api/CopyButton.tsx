"use client";

import * as React from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/utils";

interface CopyButtonProps {
  /** Raw text to copy.  For server-rendered Shiki blocks this is the
   *  original source — passing the pre-highlight string keeps the
   *  clipboard contents clean (no token-span markup). */
  text: string;
  className?: string;
}

/** Floating copy-to-clipboard control shared by every code-block-style
 *  surface in the docs site: ``ExampleBlock`` (docstring examples),
 *  ``CodeBlock`` (MDX guides), and any future code surface that needs
 *  the same affordance.  The button is invisible until the parent
 *  element is hovered / keyboard-focused. */
export function CopyButton({ text, className }: CopyButtonProps) {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = React.useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API unavailable (insecure context / old browser).
      // Fall back silently — nothing useful we can do here.
    }
  }, [text]);

  return (
    <button
      type="button"
      onClick={handleCopy}
      aria-label={copied ? "Copied" : "Copy code"}
      className={cn(
        "absolute right-3 top-3 z-10 flex h-7 w-7 items-center justify-center rounded-lg border",
        "transition-all duration-150",
        "opacity-0 group-hover:opacity-100 focus-visible:opacity-100",
        copied
          ? "border-lucid-success/40 bg-lucid-success/10 text-lucid-success"
          : "border-lucid-border bg-lucid-elevated text-lucid-text-low hover:text-lucid-text-high",
        className,
      )}
    >
      {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
    </button>
  );
}
