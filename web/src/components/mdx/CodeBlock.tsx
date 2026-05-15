"use client";

import * as React from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/utils";

interface CodeBlockProps extends React.HTMLAttributes<HTMLPreElement> {
  "data-language"?: string;
}

export function CodeBlock({
  children,
  className,
  style: _ignoredShikiStyle,
  "data-language": lang,
  ...props
}: CodeBlockProps) {
  const [copied, setCopied] = React.useState(false);
  const preRef = React.useRef<HTMLPreElement>(null);

  const handleCopy = React.useCallback(async () => {
    const text = preRef.current?.querySelector("code")?.innerText ?? "";
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, []);

  return (
    <div className="group relative my-6">
      <pre
        ref={preRef}
        className={cn(
          "overflow-x-auto rounded-xl border border-lucid-border",
          "bg-lucid-surface px-5 pt-5 pb-4 text-sm",
          "[&_code]:block [&_code]:font-mono [&_code]:text-[13px] [&_code]:leading-[1.65]",
          // shiki token spans — inherit our font
          "[&_.line]:min-h-[1.65em]",
        )}
        {...props}
      >
        {children}
      </pre>

      {/* Language badge — top-right, inside the block */}
      {lang && (
        <span
          className={cn(
            "pointer-events-none absolute right-10 top-3",
            "rounded-md px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider",
            "text-lucid-text-disabled",
          )}
        >
          {lang}
        </span>
      )}

      {/* Copy button */}
      <button
        onClick={handleCopy}
        aria-label={copied ? "Copied" : "Copy code"}
        className={cn(
          "absolute right-3 top-3 flex h-7 w-7 items-center justify-center rounded-lg border",
          "transition-all duration-150",
          "opacity-0 group-hover:opacity-100 focus-visible:opacity-100",
          copied
            ? "border-lucid-success/40 bg-lucid-success/10 text-lucid-success"
            : "border-lucid-border bg-lucid-elevated text-lucid-text-low hover:text-lucid-text-high",
        )}
      >
        {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      </button>
    </div>
  );
}

export function InlineCode({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <code
      className={cn(
        "rounded-md border border-lucid-border/70 bg-lucid-surface",
        "px-[0.4em] py-[0.15em] font-mono text-[0.84em] text-lucid-primary-light",
        className,
      )}
    >
      {children}
    </code>
  );
}
