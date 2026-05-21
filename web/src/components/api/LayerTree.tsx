"use client";

/**
 * LayerTree — torchsummary-style expandable layer view rendered from
 * the cached ``model_summary`` tree on a model-factory's JSON entry.
 *
 * Visual conventions:
 *   - Each node shows ``<name> : <Type> [× repeat]``  with param count
 *     right-aligned.
 *   - Top two levels auto-expand; deeper nodes start collapsed.
 *   - Leaf nodes (no children) render without a toggle.
 *   - Empty / param-less leaves (ReLU, MaxPool2d, ...) dim out so the
 *     eye lands on the parameterised layers first.
 *
 * Data source: ``ApiFunction.model_summary``, populated from
 * ``web/public/api-data/_summaries.json`` (built by
 * ``tools/build_model_summaries.py``).
 */

import { useState } from "react";
import { cn, formatCompactCount } from "@/lib/utils";
import type { LayerSummaryNode } from "@/lib/types";

interface LayerTreeProps {
  tree: LayerSummaryNode;
  /** Levels that start expanded by default.  0 = root only, 1 = root +
   *  children, etc.  Root is always rendered open. */
  defaultOpenDepth?: number;
}

export function LayerTree({ tree, defaultOpenDepth = 1 }: LayerTreeProps) {
  return (
    <div className="font-mono text-xs leading-relaxed">
      <LayerNode node={tree} depth={0} defaultOpenDepth={defaultOpenDepth} />
    </div>
  );
}

interface LayerNodeProps {
  node: LayerSummaryNode;
  depth: number;
  defaultOpenDepth: number;
}

function LayerNode({ node, depth, defaultOpenDepth }: LayerNodeProps) {
  const hasChildren = (node.children?.length ?? 0) > 0;
  const [open, setOpen] = useState(depth <= defaultOpenDepth);

  const hasParams = node.params > 0;
  const nameClass = hasParams
    ? "text-lucid-text-high"
    : "text-lucid-text-disabled";
  const typeClass = hasParams
    ? "text-api-fn"
    : "text-lucid-text-disabled";

  return (
    <div className={depth === 0 ? "" : "ml-4 border-l border-lucid-border/40 pl-3"}>
      <div
        className={cn(
          "flex items-baseline gap-2 py-0.5",
          hasChildren && "cursor-pointer hover:bg-lucid-elevated/40 -ml-1 pl-1 rounded",
        )}
        onClick={hasChildren ? () => setOpen((o) => !o) : undefined}
      >
        <span
          className={cn(
            "inline-block w-3 shrink-0 text-[10px] text-lucid-text-low select-none",
            !hasChildren && "opacity-0",
          )}
          aria-hidden="true"
        >
          {hasChildren ? (open ? "▾" : "▸") : "·"}
        </span>
        <span className={cn("truncate", nameClass)}>{node.name}</span>
        <span className="text-lucid-text-disabled">:</span>
        <span className={cn("truncate", typeClass)}>{node.type}</span>
        {node.repeat && (
          <span className="rounded border border-lucid-text-low/30 bg-lucid-text-low/10 px-1 text-[10px] text-lucid-text-mid">
            × {node.repeat}
          </span>
        )}
        <span
          className={cn(
            "ml-auto pl-3 tabular-nums",
            hasParams ? "text-lucid-text-mid" : "text-lucid-text-disabled",
          )}
        >
          {hasParams ? formatCompactCount(node.params) : "—"}
        </span>
      </div>
      {open && hasChildren && (
        <div>
          {node.children.map((c, i) => (
            <LayerNode
              key={`${c.name}-${i}`}
              node={c}
              depth={depth + 1}
              defaultOpenDepth={defaultOpenDepth}
            />
          ))}
        </div>
      )}
    </div>
  );
}
