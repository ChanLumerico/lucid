"use client";

import * as React from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/utils";

interface CodeBlockProps {
  children: React.ReactNode;
  className?: string;
  "data-language"?: string;
  "data-theme"?: string;
}

export function CodeBlock({ children, className, ...props }: CodeBlockProps) {
  const lang = props["data-language"];
  const [copied, setCopied] = React.useState(false);
  const preRef = React.useRef<HTMLPreElement>(null);

  const handleCopy = React.useCallback(async () => {
    const text = preRef.current?.querySelector("code")?.innerText ?? "";
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, []);

  return (
    <div className="group relative my-5">
      <pre
        ref={preRef}
        className={cn(
          "overflow-x-auto rounded-xl border border-lucid-border bg-lucid-surface px-5 py-4 text-sm",
          "[&_code]:font-mono [&_code]:text-[13px] [&_code]:leading-relaxed",
          className,
        )}
        {...props}
      >
        {children}
      </pre>

      {lang && (
        <span className="absolute left-4 top-0 -translate-y-1/2 rounded-md border border-lucid-border bg-lucid-elevated px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-lucid-text-low">
          {lang}
        </span>
      )}

      <button
        onClick={handleCopy}
        aria-label={copied ? "Copied" : "Copy code"}
        className={cn(
          "absolute right-3 top-3 flex h-7 w-7 items-center justify-center rounded-lg border transition-all duration-150",
          "opacity-0 group-hover:opacity-100 focus-visible:opacity-100",
          copied
            ? "border-lucid-success/40 bg-lucid-success/10 text-lucid-success"
            : "border-lucid-border bg-lucid-elevated text-lucid-text-low hover:text-lucid-text-high",
        )}
      >
        {copied ? (
          <Check className="h-3.5 w-3.5" />
        ) : (
          <Copy className="h-3.5 w-3.5" />
        )}
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
        "rounded-md border border-lucid-border bg-lucid-surface px-1.5 py-0.5",
        "font-mono text-[0.85em] text-lucid-primary-light",
        className,
      )}
    >
      {children}
    </code>
  );
}
