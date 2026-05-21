"use client";

/**
 * ModelSizeCard — the "MODEL SIZE" section on a factory's detail page.
 *
 * Two views collapsed into one card:
 *   - Header row (always visible) — big formatted count + exact integer.
 *     If the factory carries a layer-summary tree, a chevron on the
 *     right toggles it.
 *   - Layer tree (optional) — torchsummary-style expandable breakdown
 *     of every submodule, with consecutive identical siblings collapsed
 *     into ``Type × N`` so deeply-repetitive architectures (ResNet
 *     stages, transformer blocks) stay readable.
 *
 * Both the count and the tree come from ``@register_model`` meta info
 * (the ``params=`` kwarg + the cached output of
 * ``tools/build_model_summaries.py``), so nothing is hardcoded in
 * docstrings or here.
 */

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn, formatCompactCount } from "@/lib/utils";
import type { LayerSummaryNode } from "@/lib/types";
import { LayerTree } from "./LayerTree";

interface ModelSizeCardProps {
  paramCount?: number;
  summary?: LayerSummaryNode;
}

export function ModelSizeCard({ paramCount, summary }: ModelSizeCardProps) {
  const [open, setOpen] = useState(false);
  const hasTree = !!summary;
  // Prefer the paper-cited ``params=`` value as the headline; fall back to
  // the runtime tree total when no authoritative number was declared.
  const headlineCount = paramCount ?? summary?.params;
  if (headlineCount === undefined) return null;

  return (
    <section className="space-y-2">
      <h4 className="text-xs font-semibold tracking-widest text-lucid-text-disabled uppercase">
        Model Size
      </h4>
      <div className="rounded-xl border border-lucid-border bg-lucid-surface overflow-hidden">
        {/* Header — always visible.  Clickable when a tree is attached. */}
        <button
          type="button"
          onClick={hasTree ? () => setOpen((o) => !o) : undefined}
          disabled={!hasTree}
          className={cn(
            "w-full flex items-center gap-3 px-4 py-3 text-left",
            hasTree && "cursor-pointer hover:bg-lucid-elevated/40 transition-colors",
            !hasTree && "cursor-default",
          )}
          aria-expanded={hasTree ? open : undefined}
        >
          <span className="font-mono text-2xl font-semibold text-lucid-text-high">
            {formatCompactCount(headlineCount)}
          </span>
          <span className="font-mono text-xs text-lucid-text-low">
            {headlineCount.toLocaleString()} trainable parameters
          </span>
          {hasTree && (
            <ChevronDown
              className={cn(
                "ml-auto h-4 w-4 text-lucid-text-low transition-transform",
                open && "rotate-180",
              )}
              aria-hidden="true"
            />
          )}
        </button>
        {/* Expandable layer-tree breakdown. */}
        {hasTree && open && (
          <div className="border-t border-lucid-border px-4 py-3 max-h-[640px] overflow-y-auto">
            <LayerTree tree={summary} defaultOpenDepth={1} />
          </div>
        )}
      </div>
    </section>
  );
}
