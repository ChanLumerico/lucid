import type { MDXComponents } from "mdx/types";
import Link from "next/link";
import { CodeBlock, InlineCode } from "./CodeBlock";
import { Callout } from "./Callout";
import { Tabs, Tab } from "./Tabs";
import { cn } from "@/lib/utils";

export { Callout, Tabs, Tab, CodeBlock, InlineCode };

export function getMDXComponents(
  overrides: MDXComponents = {},
): MDXComponents {
  return {
    h1: ({ children, ...props }) => (
      <h1
        {...props}
        className="mt-10 mb-4 scroll-mt-20 text-3xl font-bold tracking-tight text-lucid-text-high first:mt-0"
      >
        {children}
      </h1>
    ),
    h2: ({ children, id, ...props }) => (
      <h2
        {...props}
        id={id}
        className="group mt-10 mb-3 scroll-mt-20 text-xl font-semibold text-lucid-text-high"
      >
        <a
          href={id ? `#${id}` : undefined}
          className="no-underline hover:text-lucid-primary transition-colors"
        >
          {children}
        </a>
      </h2>
    ),
    h3: ({ children, id, ...props }) => (
      <h3
        {...props}
        id={id}
        className="mt-8 mb-2 scroll-mt-20 text-base font-semibold text-lucid-text-high"
      >
        <a
          href={id ? `#${id}` : undefined}
          className="no-underline hover:text-lucid-primary transition-colors"
        >
          {children}
        </a>
      </h3>
    ),
    p: ({ children, ...props }) => (
      <p
        {...props}
        className="my-4 leading-7 text-lucid-text-mid"
      >
        {children}
      </p>
    ),
    a: ({ href, children, ...props }) => {
      const isExternal = href?.startsWith("http");
      if (isExternal) {
        return (
          <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            {...props}
            className="text-lucid-primary underline underline-offset-4 decoration-lucid-primary/40 hover:decoration-lucid-primary transition-colors"
          >
            {children}
          </a>
        );
      }
      return (
        <Link
          href={href ?? "#"}
          {...props}
          className="text-lucid-primary underline underline-offset-4 decoration-lucid-primary/40 hover:decoration-lucid-primary transition-colors"
        >
          {children}
        </Link>
      );
    },
    code: ({ children, className, ...props }) => {
      const isBlock = className?.includes("language-");
      if (isBlock) {
        return (
          <code className={cn("block", className)} {...props}>
            {children}
          </code>
        );
      }
      return <InlineCode className={className}>{children}</InlineCode>;
    },
    pre: ({ children, ...props }) => (
      <CodeBlock {...(props as React.HTMLAttributes<HTMLPreElement>)}>
        {children}
      </CodeBlock>
    ),
    ul: ({ children, ...props }) => (
      <ul
        {...props}
        className="my-4 ml-5 list-disc space-y-1.5 text-lucid-text-mid marker:text-lucid-text-disabled"
      >
        {children}
      </ul>
    ),
    ol: ({ children, ...props }) => (
      <ol
        {...props}
        className="my-4 ml-5 list-decimal space-y-1.5 text-lucid-text-mid marker:text-lucid-text-disabled"
      >
        {children}
      </ol>
    ),
    li: ({ children, ...props }) => (
      <li {...props} className="leading-7">
        {children}
      </li>
    ),
    blockquote: ({ children, ...props }) => (
      <blockquote
        {...props}
        className="my-5 border-l-2 border-lucid-primary/50 pl-4 text-lucid-text-low italic"
      >
        {children}
      </blockquote>
    ),
    hr: () => <hr className="my-8 border-lucid-border" />,
    table: ({ children, ...props }) => (
      <div className="my-6 overflow-x-auto rounded-xl border border-lucid-border">
        <table
          {...props}
          className="w-full text-sm text-lucid-text-mid border-collapse"
        >
          {children}
        </table>
      </div>
    ),
    thead: ({ children, ...props }) => (
      <thead
        {...props}
        className="border-b border-lucid-border bg-lucid-surface"
      >
        {children}
      </thead>
    ),
    th: ({ children, ...props }) => (
      <th
        {...props}
        className="px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wider text-lucid-text-low"
      >
        {children}
      </th>
    ),
    td: ({ children, ...props }) => (
      <td
        {...props}
        className="border-t border-lucid-border/50 px-4 py-2.5"
      >
        {children}
      </td>
    ),
    Callout,
    Tabs,
    Tab,
    ...overrides,
  };
}
