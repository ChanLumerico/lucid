import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { cn } from "@/lib/utils";
import type { Components } from "react-markdown";

const components: Components = {
  p: ({ children }) => (
    <p className="leading-relaxed">{children}</p>
  ),
  strong: ({ children }) => (
    <strong className="font-semibold text-lucid-text-high">{children}</strong>
  ),
  em: ({ children }) => (
    <em className="italic">{children}</em>
  ),
  code: ({ children, className }) => {
    if (className) {
      return (
        <code className={cn("font-mono text-sm", className)}>{children}</code>
      );
    }
    return (
      <code className="font-mono text-[0.875em] bg-lucid-elevated text-lucid-primary-light px-1 py-px rounded">
        {children}
      </code>
    );
  },
  pre: ({ children }) => (
    <pre className="rounded-lg border border-lucid-border bg-lucid-elevated px-3 py-2.5 overflow-x-auto text-xs font-mono leading-relaxed my-2">
      {children}
    </pre>
  ),
  ul: ({ children }) => (
    <ul className="list-none space-y-1 pl-1 my-1">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="list-decimal pl-5 space-y-1 my-1">{children}</ol>
  ),
  li: ({ children }) => (
    <li className="flex items-start gap-2 text-sm">
      <span className="mt-[0.4em] h-1 w-1 shrink-0 rounded-full bg-lucid-text-disabled" />
      <span className="flex-1">{children}</span>
    </li>
  ),
  a: ({ href, children }) => (
    <a
      href={href}
      className="text-lucid-primary-light underline underline-offset-2 hover:text-lucid-primary transition-colors"
      target="_blank"
      rel="noopener noreferrer"
    >
      {children}
    </a>
  ),
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-lucid-blue bg-lucid-blue/5 px-4 py-2 my-2 text-lucid-text-mid">
      {children}
    </blockquote>
  ),
  h1: ({ children }) => (
    <h1 className="text-base font-bold text-lucid-text-high mt-3 mb-1">{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 className="text-sm font-semibold text-lucid-text-high mt-3 mb-1">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="text-sm font-medium text-lucid-text-high mt-2 mb-0.5">{children}</h3>
  ),
};

const PLUGINS = {
  remark: [remarkGfm, remarkMath] as Parameters<typeof Markdown>[0]["remarkPlugins"],
  rehype: [rehypeKatex] as Parameters<typeof Markdown>[0]["rehypePlugins"],
};

interface MathTextProps {
  text: string;
  className?: string;
  block?: boolean;
}

export function MathText({ text, className, block = false }: MathTextProps) {
  if (!text) return null;

  if (!block) {
    return (
      <span className={cn("inline text-sm leading-relaxed", className)}>
        <Markdown
          remarkPlugins={PLUGINS.remark}
          rehypePlugins={PLUGINS.rehype}
          components={{
            ...components,
            p: ({ children }) => <span>{children}</span>,
          }}
        >
          {text}
        </Markdown>
      </span>
    );
  }

  return (
    <div
      className={cn(
        "text-sm leading-relaxed space-y-2",
        "[&>*:first-child]:mt-0 [&>*:last-child]:mb-0",
        className,
      )}
    >
      <Markdown
        remarkPlugins={PLUGINS.remark}
        rehypePlugins={PLUGINS.rehype}
        components={components}
      >
        {text}
      </Markdown>
    </div>
  );
}
