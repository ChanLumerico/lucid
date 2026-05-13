import fs from "fs";
import path from "path";
import type { Metadata } from "next";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { FadeIn } from "@/components/motion/FadeIn";
import { InlineCode } from "@/components/mdx/CodeBlock";
import { cn } from "@/lib/utils";

export const metadata: Metadata = {
  title: "Changelog",
  description: "What's new in Lucid.",
};

// ── Parser ────────────────────────────────────────────────────────────────────

interface ChangelogCategory {
  name: string;
  items: string[];
}

interface ChangelogVersion {
  version: string;
  date: string | null;
  isUnreleased: boolean;
  isPreRelease: boolean;
  description: string;
  categories: ChangelogCategory[];
}

function parseChangelog(raw: string): ChangelogVersion[] {
  const content = raw.replace(/^\[[^\]]+\]:\s*https?:\/\/.+$/gm, "").trim();
  const sections = content.split(/^(?=## )/m).filter((s) => s.startsWith("## "));

  return sections.map((section) => {
    const lines = section.split("\n");
    const headerLine = lines[0].replace(/^## /, "").trim();

    const headerMatch = headerLine.match(/^\[?([^\]\n]+)\]?(?:\s*[—–-]+\s*(.+))?/);
    const version = headerMatch?.[1]?.trim() ?? headerLine;
    const date = headerMatch?.[2]?.trim() ?? null;
    const isUnreleased = version.toLowerCase() === "unreleased";
    const isPreRelease = version.toLowerCase().startsWith("pre");

    const body = lines.slice(1).join("\n");
    const catParts = body.split(/^### /m);

    const description = catParts[0]
      .replace(/^---+$/m, "")
      .trim();

    const categories: ChangelogCategory[] = [];
    for (const catPart of catParts.slice(1)) {
      const catLines = catPart.split("\n");
      const catName = catLines[0].trim();
      const items: string[] = [];
      let current = "";

      for (const line of catLines.slice(1)) {
        if (line.startsWith("- ")) {
          if (current) items.push(current.trim());
          current = line.slice(2);
        } else if (current && line.trim() && !line.startsWith("#")) {
          current += " " + line.trim();
        }
      }
      if (current) items.push(current.trim());
      if (items.length > 0) categories.push({ name: catName, items });
    }

    return { version, date, isUnreleased, isPreRelease, description, categories };
  });
}

// ── Inline markdown renderer ──────────────────────────────────────────────────

function renderInline(text: string): React.ReactNode[] {
  const pattern = /(\*\*`[^`]+`\*\*|\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g;
  const parts = text.split(pattern);

  return parts.map((part, i) => {
    // **`code`** — bold code
    if (/^\*\*`[^`]+`\*\*$/.test(part)) {
      const inner = part.slice(3, -3);
      return (
        <strong key={i} className="font-semibold text-lucid-primary-light">
          <InlineCode>{inner}</InlineCode>
        </strong>
      );
    }
    // **bold**
    if (/^\*\*[^*]+\*\*$/.test(part)) {
      return (
        <strong key={i} className="font-semibold text-lucid-text-high">
          {part.slice(2, -2)}
        </strong>
      );
    }
    // `code`
    if (/^`[^`]+`$/.test(part)) {
      return <InlineCode key={i}>{part.slice(1, -1)}</InlineCode>;
    }
    // [text](url)
    const linkMatch = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (linkMatch) {
      return (
        <a
          key={i}
          href={linkMatch[2]}
          target="_blank"
          rel="noopener noreferrer"
          className="text-lucid-primary underline underline-offset-4 decoration-lucid-primary/40"
        >
          {linkMatch[1]}
        </a>
      );
    }
    return part;
  });
}

// ── Category label styles ─────────────────────────────────────────────────────

const CATEGORY_COLOR: Record<string, string> = {
  added:         "text-lucid-success",
  fixed:         "text-lucid-warning",
  changed:       "text-lucid-blue",
  refactored:    "text-lucid-blue",
  removed:       "text-lucid-error",
  deprecated:    "text-lucid-warning",
  performance:   "text-lucid-primary",
  security:      "text-lucid-error",
  tooling:       "text-lucid-text-low",
  documentation: "text-lucid-text-low",
  breaking:      "text-lucid-error",
};

function categoryColor(name: string): string {
  const key = name.toLowerCase().split(/[\s—–]/)[0];
  return CATEGORY_COLOR[key] ?? "text-lucid-text-low";
}

// ── Components ────────────────────────────────────────────────────────────────

function VersionSection({ entry }: { entry: ChangelogVersion }) {
  const { version, date, isUnreleased, isPreRelease, description, categories } = entry;

  const dotClass = isUnreleased
    ? "bg-lucid-primary ring-lucid-bg"
    : isPreRelease
      ? "bg-lucid-text-disabled ring-lucid-bg"
      : "bg-lucid-blue ring-lucid-bg";

  const versionLabel = isPreRelease
    ? version
    : isUnreleased
      ? "Unreleased"
      : `v${version}`;

  return (
    <div className="relative pl-7 pb-12 last:pb-0">
      {/* Timeline dot */}
      <span
        className={cn(
          "absolute left-[-4px] top-[6px] h-2.5 w-2.5 rounded-full ring-2",
          dotClass,
          isUnreleased && "ring-offset-1 ring-offset-lucid-bg animate-pulse",
        )}
      />

      {/* Version header */}
      <div className="flex flex-wrap items-baseline gap-3 mb-3">
        <h2 className="text-base font-bold text-lucid-text-high">
          {versionLabel}
        </h2>
        {isUnreleased && (
          <span className="rounded-full border border-lucid-primary/30 bg-lucid-primary/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-lucid-primary">
            Unreleased
          </span>
        )}
        {date && (
          <time className="text-xs text-lucid-text-disabled font-mono">
            {date}
          </time>
        )}
      </div>

      {/* Release description */}
      {description && (
        <p className="mb-4 text-sm text-lucid-text-low leading-relaxed max-w-xl">
          {description}
        </p>
      )}

      {/* Categories */}
      {categories.map((cat) => (
        <div key={cat.name} className="mb-4">
          <p
            className={cn(
              "mb-1.5 text-[10px] font-bold uppercase tracking-widest",
              categoryColor(cat.name),
            )}
          >
            {cat.name}
          </p>
          <ul className="space-y-1">
            {cat.items.map((item, i) => (
              <li
                key={i}
                className="flex items-start gap-2 text-[13px] text-lucid-text-mid leading-relaxed"
              >
                <span className="mt-[6px] h-1 w-1 shrink-0 rounded-full bg-lucid-border" />
                <span>{renderInline(item)}</span>
              </li>
            ))}
          </ul>
        </div>
      ))}

      {/* Pre-release text only */}
      {isPreRelease && categories.length === 0 && !description && (
        <p className="text-sm text-lucid-text-disabled italic">
          No formal changelog — see git history.
        </p>
      )}
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default async function ChangelogPage() {
  const filePath = path.join(process.cwd(), "..", "CHANGELOG.md");
  const raw = fs.existsSync(filePath) ? fs.readFileSync(filePath, "utf-8") : "";
  const entries = parseChangelog(raw);

  return (
    <div className="flex min-h-dvh flex-col">
      <Header />
      <main className="flex-1 pt-14">
        <div className="mx-auto max-w-2xl px-4 sm:px-6 py-12">
          <FadeIn>
            <h1 className="text-3xl font-bold text-lucid-text-high mb-1">
              Changelog
            </h1>
            <p className="text-lucid-text-low mb-10 text-sm">
              All notable changes to Lucid.{" "}
              <a
                href="https://keepachangelog.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-lucid-primary/80 hover:text-lucid-primary underline underline-offset-4 transition-colors"
              >
                Keep a Changelog
              </a>{" "}
              format.
            </p>

            <div className="relative before:absolute before:left-[-0.5px] before:top-2 before:bottom-2 before:w-px before:bg-lucid-border">
              {entries.map((entry) => (
                <VersionSection key={entry.version} entry={entry} />
              ))}
            </div>
          </FadeIn>
        </div>
      </main>
      <Footer />
    </div>
  );
}
