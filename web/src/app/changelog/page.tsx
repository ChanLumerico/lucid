import fs from "fs";
import path from "path";
import type { Metadata } from "next";
import { compileMDX } from "next-mdx-remote/rsc";
import remarkGfm from "remark-gfm";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { FadeIn } from "@/components/motion/FadeIn";
import { Badge } from "@/components/ui/badge";
import { InlineCode } from "@/components/mdx/CodeBlock";
import { cn } from "@/lib/utils";

export const metadata: Metadata = {
  title: "Changelog",
  description: "What's new in Lucid.",
};

// ── Changelog-specific MDX components ────────────────────────────────────────

function VersionHeading({ children }: { children: React.ReactNode }) {
  const text = String(children).trim();

  // "[Unreleased]" or "[3.0.0] — 2026-05-10"
  const match = text.match(/^\[?([^\]]+)\]?(?:\s*[—–-]+\s*(.+))?$/);
  const version = match?.[1]?.trim() ?? text;
  const date = match?.[2]?.trim();

  const isUnreleased = version.toLowerCase() === "unreleased";
  const isPreRelease = version.toLowerCase().startsWith("pre");

  return (
    <div className="relative pl-6 mt-10 first:mt-0">
      {/* Timeline dot */}
      <div
        className={cn(
          "absolute left-[-3.5px] top-1.5 h-2.5 w-2.5 rounded-full ring-2 ring-lucid-bg",
          isUnreleased
            ? "bg-lucid-primary"
            : isPreRelease
              ? "bg-lucid-text-disabled"
              : "bg-lucid-blue",
        )}
      />

      <div className="flex flex-wrap items-center gap-2 mb-4">
        <span className="text-lg font-bold text-lucid-text-high">
          {isPreRelease ? version : `v${version}`}
        </span>
        {isUnreleased && <Badge variant="default">Unreleased</Badge>}
        {date && (
          <time className="text-xs text-lucid-text-disabled">{date}</time>
        )}
      </div>
    </div>
  );
}

function CategoryHeading({ children }: { children: React.ReactNode }) {
  return (
    <p className="mt-5 mb-2 pl-6 text-[11px] font-semibold uppercase tracking-widest text-lucid-text-low">
      {children}
    </p>
  );
}

function ChangelogList({ children }: { children: React.ReactNode }) {
  return (
    <ul className="pl-6 space-y-1.5">
      {children}
    </ul>
  );
}

function ChangelogItem({ children }: { children: React.ReactNode }) {
  return (
    <li className="flex items-start gap-2 text-sm text-lucid-text-mid leading-relaxed">
      <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-lucid-border" />
      <span>{children}</span>
    </li>
  );
}

function ChangelogParagraph({ children }: { children: React.ReactNode }) {
  return (
    <p className="pl-6 mt-2 mb-3 text-sm text-lucid-text-low leading-relaxed italic">
      {children}
    </p>
  );
}

function ChangelogLink({
  href,
  children,
}: {
  href?: string;
  children: React.ReactNode;
}) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-lucid-primary underline underline-offset-4 decoration-lucid-primary/40 hover:decoration-lucid-primary transition-colors"
    >
      {children}
    </a>
  );
}

const CHANGELOG_COMPONENTS = {
  h2: VersionHeading,
  h3: CategoryHeading,
  ul: ChangelogList,
  li: ChangelogItem,
  p: ChangelogParagraph,
  a: ChangelogLink,
  code: InlineCode,
};

// ── Page ─────────────────────────────────────────────────────────────────────

async function getChangelog() {
  // CHANGELOG.md is at repo root — one level above web/
  const filePath = path.join(process.cwd(), "..", "CHANGELOG.md");
  if (!fs.existsSync(filePath)) return null;

  // Strip the reference-link footer (lines like "[3.0.0]: https://...")
  // so remark doesn't turn version headers into anchor tags
  const raw = fs.readFileSync(filePath, "utf-8");
  const stripped = raw.replace(/^\[[^\]]+\]:\s*https?:\/\/.+$/gm, "");

  const { content } = await compileMDX({
    source: stripped,
    components: CHANGELOG_COMPONENTS,
    options: {
      mdxOptions: {
        format: "md",       // pure markdown — prevents {…} being parsed as JSX expressions
        remarkPlugins: [remarkGfm],
      },
    },
  });

  return content;
}

export default async function ChangelogPage() {
  const content = await getChangelog();

  return (
    <div className="flex min-h-dvh flex-col">
      <Header />
      <main className="flex-1 pt-14">
        <div className="mx-auto max-w-2xl px-4 sm:px-6 py-12">
          <FadeIn>
            <h1 className="text-3xl font-bold text-lucid-text-high mb-2">
              Changelog
            </h1>
            <p className="text-lucid-text-mid mb-12 text-sm leading-relaxed">
              All notable changes to Lucid, based on{" "}
              <a
                href="https://keepachangelog.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-lucid-primary underline underline-offset-4"
              >
                Keep a Changelog
              </a>
              .
            </p>

            {content ? (
              <div className="relative before:absolute before:left-0 before:top-3 before:bottom-3 before:w-px before:bg-lucid-border">
                {content}
              </div>
            ) : (
              <p className="text-lucid-text-low text-sm">
                CHANGELOG.md not found.
              </p>
            )}
          </FadeIn>
        </div>
      </main>
      <Footer />
    </div>
  );
}
