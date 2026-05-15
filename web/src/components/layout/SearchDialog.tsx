"use client";

import * as React from "react";
import Fuse from "fuse.js";
import { Search, FileText, Box, Zap, BookOpen } from "lucide-react";
import { useRouter } from "next/navigation";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import type { SearchEntry } from "@/lib/search-index";

const KIND_CONFIG = {
  "api-module": {
    icon: Box,
    color: "text-lucid-primary",
    bg: "bg-lucid-primary/10",
    label: "Module",
  },
  "api-class": {
    icon: Box,
    color: "text-lucid-warning",
    bg: "bg-lucid-warning/10",
    label: "Class",
  },
  "api-function": {
    icon: Zap,
    color: "text-lucid-blue",
    bg: "bg-lucid-blue/10",
    label: "Function",
  },
  doc: {
    icon: BookOpen,
    color: "text-lucid-success",
    bg: "bg-lucid-success/10",
    label: "Guide",
  },
} as const;

function KindIcon({ kind }: { kind: SearchEntry["kind"] }) {
  const { icon: Icon, color, bg } = KIND_CONFIG[kind];
  return (
    <span
      className={cn(
        "flex h-7 w-7 shrink-0 items-center justify-center rounded-lg",
        bg,
      )}
    >
      <Icon className={cn("h-3.5 w-3.5", color)} />
    </span>
  );
}

interface SearchResultItemProps {
  entry: SearchEntry;
  active: boolean;
  onSelect: () => void;
}

function SearchResultItem({ entry, active, onSelect }: SearchResultItemProps) {
  return (
    <button
      onClick={onSelect}
      className={cn(
        "flex w-full items-start gap-3 rounded-xl px-3 py-2.5 text-left transition-colors",
        active ? "bg-lucid-elevated" : "hover:bg-lucid-elevated/60",
      )}
    >
      <KindIcon kind={entry.kind} />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-lucid-text-high truncate">
            {entry.title}
          </span>
          {entry.badge && (
            <span className="shrink-0 text-[10px] text-lucid-text-disabled">
              {entry.badge}
            </span>
          )}
        </div>
        {entry.summary && (
          <p className="mt-0.5 text-xs text-lucid-text-low line-clamp-1">
            {entry.summary}
          </p>
        )}
      </div>
    </button>
  );
}

export function SearchDialog({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const router = useRouter();
  const [query, setQuery] = React.useState("");
  const [results, setResults] = React.useState<SearchEntry[]>([]);
  const [activeIndex, setActiveIndex] = React.useState(0);
  const [index, setIndex] = React.useState<Fuse<SearchEntry> | null>(null);
  const inputRef = React.useRef<HTMLInputElement>(null);

  // Load search index once
  React.useEffect(() => {
    if (!open) return;
    if (index) return;
    fetch("/api/search-data")
      .then((r) => r.json())
      .then((data: SearchEntry[]) => {
        setIndex(
          new Fuse(data, {
            keys: [
              { name: "title", weight: 3 },
              { name: "summary", weight: 1 },
              { name: "badge", weight: 1 },
            ],
            threshold: 0.35,
            includeScore: true,
          }),
        );
      })
      .catch(() => {});
  }, [open, index]);

  // Focus input on open
  React.useEffect(() => {
    if (open) {
      setQuery("");
      setActiveIndex(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [open]);

  // Run search
  React.useEffect(() => {
    if (!index || !query.trim()) {
      setResults([]);
      setActiveIndex(0);
      return;
    }
    const hits = index.search(query, { limit: 8 });
    setResults(hits.map((h) => h.item));
    setActiveIndex(0);
  }, [query, index]);

  const navigate = React.useCallback(
    (entry: SearchEntry) => {
      router.push(entry.href);
      onClose();
    },
    [router, onClose],
  );

  const handleKeyDown = React.useCallback(
    (e: React.KeyboardEvent) => {
      if (results.length === 0) return;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActiveIndex((i) => Math.min(i + 1, results.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setActiveIndex((i) => Math.max(i - 1, 0));
      } else if (e.key === "Enter") {
        e.preventDefault();
        navigate(results[activeIndex]);
      }
    },
    [results, activeIndex, navigate],
  );

  return (
    <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
      <DialogContent>
        {/* Search input */}
        <div className="flex items-center gap-3 border-b border-lucid-border px-4 py-3.5">
          <Search className="h-4 w-4 shrink-0 text-lucid-text-low" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search docs and API…"
            className="flex-1 bg-transparent text-sm text-lucid-text-high placeholder:text-lucid-text-disabled outline-none"
          />
          <kbd className="hidden sm:inline-flex items-center gap-0.5 rounded border border-lucid-border px-1.5 py-0.5 text-[10px] text-lucid-text-disabled">
            esc
          </kbd>
        </div>

        {/* Results */}
        <div className="max-h-[360px] overflow-y-auto p-2">
          {query.trim() === "" && (
            <div className="flex flex-col items-center gap-2 py-10">
              <FileText className="h-8 w-8 text-lucid-text-disabled" />
              <p className="text-sm text-lucid-text-low">
                Start typing to search…
              </p>
            </div>
          )}

          {query.trim() !== "" && results.length === 0 && (
            <div className="py-10 text-center">
              <p className="text-sm text-lucid-text-low">
                No results for{" "}
                <span className="font-medium text-lucid-text-mid">
                  &ldquo;{query}&rdquo;
                </span>
              </p>
            </div>
          )}

          {results.map((entry, i) => (
            <SearchResultItem
              key={entry.id}
              entry={entry}
              active={i === activeIndex}
              onSelect={() => navigate(entry)}
            />
          ))}
        </div>

        {/* Footer hint */}
        {results.length > 0 && (
          <div className="flex items-center gap-3 border-t border-lucid-border px-4 py-2.5 text-[11px] text-lucid-text-disabled">
            <span>
              <kbd className="rounded border border-lucid-border px-1 py-0.5">↑</kbd>
              <kbd className="ml-0.5 rounded border border-lucid-border px-1 py-0.5">↓</kbd>{" "}
              navigate
            </span>
            <span>
              <kbd className="rounded border border-lucid-border px-1 py-0.5">↵</kbd>{" "}
              open
            </span>
            <span>
              <kbd className="rounded border border-lucid-border px-1 py-0.5">esc</kbd>{" "}
              close
            </span>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
