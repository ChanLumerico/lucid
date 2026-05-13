"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown } from "lucide-react";
import { InlineCode } from "@/components/mdx/CodeBlock";
import { cn } from "@/lib/utils";
import { springs } from "@/components/motion/springs";

// ── Types (must match server-side ChangelogVersion) ───────────────────────────

export interface ChangelogCategory {
  name: string;
  items: string[];
}

export interface ChangelogVersion {
  version: string;
  date: string | null;
  isUnreleased: boolean;
  isPreRelease: boolean;
  description: string;
  categories: ChangelogCategory[];
}

// ── Inline markdown renderer ──────────────────────────────────────────────────

function renderInline(text: string): React.ReactNode[] {
  const pattern = /(\*\*`[^`]+`\*\*|\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g;
  return text.split(pattern).map((part, i) => {
    if (/^\*\*`[^`]+`\*\*$/.test(part))
      return <strong key={i} className="font-semibold text-lucid-primary-light"><InlineCode>{part.slice(3, -3)}</InlineCode></strong>;
    if (/^\*\*[^*]+\*\*$/.test(part))
      return <strong key={i} className="font-semibold text-lucid-text-high">{part.slice(2, -2)}</strong>;
    if (/^`[^`]+`$/.test(part))
      return <InlineCode key={i}>{part.slice(1, -1)}</InlineCode>;
    const link = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (link)
      return <a key={i} href={link[2]} target="_blank" rel="noopener noreferrer" className="text-lucid-primary underline underline-offset-4 decoration-lucid-primary/40">{link[1]}</a>;
    return part;
  });
}

// ── Category color ────────────────────────────────────────────────────────────

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

function categoryColor(name: string) {
  const key = name.toLowerCase().split(/[\s—–]/)[0];
  return CATEGORY_COLOR[key] ?? "text-lucid-text-low";
}

// ── Collapsed summary ─────────────────────────────────────────────────────────

function CollapsedSummary({ categories }: { categories: ChangelogCategory[] }) {
  const total = categories.reduce((s, c) => s + c.items.length, 0);
  const topCats = [...new Set(
    categories.map((c) => c.name.split(/[\s—–]/)[0])
  )].slice(0, 4);

  if (total === 0) return null;

  return (
    <div className="flex flex-wrap items-center gap-1.5 mt-2">
      {topCats.map((cat) => (
        <span
          key={cat}
          className={cn(
            "text-[10px] font-semibold uppercase tracking-wider",
            categoryColor(cat),
          )}
        >
          {cat}
        </span>
      ))}
      {categories.length > 4 && (
        <span className="text-[10px] text-lucid-text-disabled">
          +{categories.length - 4} more
        </span>
      )}
      <span className="text-[10px] text-lucid-text-disabled ml-1">
        · {total} {total === 1 ? "change" : "changes"}
      </span>
    </div>
  );
}

// ── Single version section ────────────────────────────────────────────────────

function VersionSection({
  entry,
  isExpanded,
  onToggle,
}: {
  entry: ChangelogVersion;
  isExpanded: boolean;
  onToggle: () => void;
}) {
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

  const hasContent = categories.length > 0 || description;

  return (
    <div className="relative pl-7 pb-10 last:pb-0">
      {/* Timeline dot */}
      <span
        className={cn(
          "absolute left-[-4px] top-[7px] h-2.5 w-2.5 rounded-full ring-2",
          dotClass,
          isUnreleased && "animate-pulse",
        )}
      />

      {/* Header row — clickable */}
      <button
        onClick={onToggle}
        disabled={!hasContent}
        className={cn(
          "group w-full text-left",
          hasContent && "cursor-pointer",
        )}
        aria-expanded={isExpanded}
      >
        <div className="flex items-center justify-between gap-3">
          <div className="flex flex-wrap items-baseline gap-2.5">
            <span className="text-base font-bold text-lucid-text-high">
              {versionLabel}
            </span>
            {isUnreleased && (
              <span className="rounded-full border border-lucid-primary/30 bg-lucid-primary/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-lucid-primary">
                Unreleased
              </span>
            )}
            {date && (
              <time className="font-mono text-xs text-lucid-text-disabled">
                {date}
              </time>
            )}
          </div>

          {hasContent && (
            <motion.span
              animate={{ rotate: isExpanded ? 180 : 0 }}
              transition={springs.micro}
              className="shrink-0 text-lucid-text-disabled group-hover:text-lucid-text-low transition-colors"
            >
              <ChevronDown className="h-4 w-4" />
            </motion.span>
          )}
        </div>

        {/* Collapsed summary */}
        {!isExpanded && <CollapsedSummary categories={categories} />}
      </button>

      {/* Expandable content */}
      <AnimatePresence initial={false}>
        {isExpanded && (
          <motion.div
            key="content"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: [0.4, 0, 0.2, 1] }}
            style={{ overflow: "hidden" }}
          >
            <div className="pt-3">
              {description && (
                <p className="mb-4 max-w-xl text-sm leading-relaxed text-lucid-text-low">
                  {description}
                </p>
              )}

              {categories.map((cat) => (
                <div key={cat.name} className="mb-4">
                  <p className={cn("mb-1.5 text-[10px] font-bold uppercase tracking-widest", categoryColor(cat.name))}>
                    {cat.name}
                  </p>
                  <ul className="space-y-1">
                    {cat.items.map((item, i) => (
                      <li key={i} className="flex items-start gap-2 text-[13px] leading-relaxed text-lucid-text-mid">
                        <span className="mt-[6px] h-1 w-1 shrink-0 rounded-full bg-lucid-border" />
                        <span>{renderInline(item)}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── Accordion ─────────────────────────────────────────────────────────────────

export function ChangelogAccordion({ entries }: { entries: ChangelogVersion[] }) {
  // Default: first stable release expanded, rest collapsed
  const firstStable = entries.find((e) => !e.isUnreleased && !e.isPreRelease);
  const [expanded, setExpanded] = React.useState<Set<string>>(
    new Set(firstStable ? [firstStable.version] : []),
  );

  const toggle = (version: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(version)) next.delete(version);
      else next.add(version);
      return next;
    });
  };

  return (
    <div className="relative before:absolute before:left-[-0.5px] before:top-2 before:bottom-2 before:w-px before:bg-lucid-border">
      {entries.map((entry) => (
        <VersionSection
          key={entry.version}
          entry={entry}
          isExpanded={expanded.has(entry.version)}
          onToggle={() => toggle(entry.version)}
        />
      ))}
    </div>
  );
}
