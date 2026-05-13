"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { ScrollArea } from "@/components/ui/scroll-area";
import { springs } from "@/components/motion/springs";

export interface SidebarItem {
  title: string;
  href?: string;
  items?: SidebarItem[];
  /** "C" = class, "f" = function, or a plain count string like "22" */
  badge?: string;
}

// ---------------------------------------------------------------------------
// KindBadge — renders "C"/"f" with semantic api colors, or a plain count
// ---------------------------------------------------------------------------
function KindBadge({
  badge,
  depth,
  active,
}: {
  badge: string;
  depth: number;
  active: boolean;
}) {
  // Leaf-level kind indicators ("C" = class, "f" = function/fn)
  if (badge === "C") {
    return (
      <span className="rounded px-1 py-px text-[8px] font-bold tracking-wide border bg-api-class/10 border-api-class/30 text-api-class">
        C
      </span>
    );
  }
  if (badge === "f") {
    return (
      <span className="rounded px-1 py-px text-[8px] font-bold tracking-wide border bg-api-fn/10 border-api-fn/30 text-api-fn">
        f
      </span>
    );
  }
  // Plain count badge (module level)
  return (
    <span
      className={cn(
        "rounded-sm px-1.5 py-0.5 text-[10px] font-semibold",
        active
          ? "bg-api-class/15 text-api-class"
          : "bg-lucid-surface text-lucid-text-disabled",
      )}
    >
      {badge}
    </span>
  );
}

interface SidebarGroupProps {
  item: SidebarItem;
  depth?: number;
}

function SidebarGroup({ item, depth = 0 }: SidebarGroupProps) {
  const pathname = usePathname();
  const isActive = item.href ? pathname === item.href || pathname.startsWith(`${item.href}/`) : false;
  const hasChildren = Boolean(item.items?.length);

  const isChildActive = React.useMemo(() => {
    if (!item.items) return false;
    return item.items.some(
      (child) => child.href && (pathname === child.href || pathname.startsWith(`${child.href}/`)),
    );
  }, [item.items, pathname]);

  const [expanded, setExpanded] = React.useState(isActive || isChildActive);

  React.useEffect(() => {
    if (isChildActive) setExpanded(true);
  }, [isChildActive]);

  if (!hasChildren && item.href) {
    return (
      <Link
        href={item.href}
        className={cn(
          "group flex items-center justify-between rounded-md px-2.5 py-1.5 text-sm transition-colors",
          depth === 0 ? "font-medium" : "font-normal",
          isActive
            ? "bg-lucid-primary/10 text-lucid-primary"
            : "text-lucid-text-low hover:bg-lucid-surface hover:text-lucid-text-mid",
        )}
        aria-current={isActive ? "page" : undefined}
        style={{ paddingLeft: `${0.625 + depth * 0.75}rem` }}
      >
        <span className="truncate">{item.title}</span>
        {item.badge && (
          <KindBadge badge={item.badge} depth={depth} active={isActive} />
        )}
      </Link>
    );
  }

  return (
    <div>
      {/* Row: navigable link + separate expand chevron */}
      <div
        className={cn(
          "group flex items-center rounded-md text-sm transition-colors",
          depth === 0 ? "font-medium" : "font-normal",
          isActive || isChildActive
            ? "text-lucid-text-high"
            : "text-lucid-text-low hover:text-lucid-text-mid",
        )}
      >
        {/* Title area — navigates if href exists */}
        {item.href ? (
          <Link
            href={item.href}
            onClick={() => setExpanded((v) => !v)}
            className={cn(
              "flex flex-1 items-center gap-1.5 rounded-md px-2.5 py-1.5 transition-colors min-w-0",
              isActive || isChildActive
                ? "hover:bg-lucid-primary/10"
                : "hover:bg-lucid-surface",
            )}
            style={{ paddingLeft: `${0.625 + depth * 0.75}rem` }}
            aria-current={isActive ? "page" : undefined}
          >
            <span className="truncate">{item.title}</span>
            {item.badge && (
              <KindBadge badge={item.badge} depth={depth} active={isActive || isChildActive} />
            )}
          </Link>
        ) : (
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className={cn(
              "flex flex-1 items-center gap-1.5 rounded-md px-2.5 py-1.5 transition-colors min-w-0",
              isActive || isChildActive ? "hover:bg-lucid-primary/10" : "hover:bg-lucid-surface",
            )}
            style={{ paddingLeft: `${0.625 + depth * 0.75}rem` }}
          >
            <span className="truncate">{item.title}</span>
            {item.badge && (
              <KindBadge badge={item.badge} depth={depth} active={isActive || isChildActive} />
            )}
          </button>
        )}

        {/* Chevron — always toggles expansion */}
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="shrink-0 rounded-md p-1.5 hover:bg-lucid-surface transition-colors"
          aria-expanded={expanded}
          aria-label={expanded ? "Collapse" : "Expand"}
        >
          <motion.span
            animate={{ rotate: expanded ? 90 : 0 }}
            transition={springs.micro}
            className="block"
          >
            <ChevronRight className="h-3.5 w-3.5 text-lucid-text-disabled" />
          </motion.span>
        </button>
      </div>

      <AnimatePresence initial={false}>
        {expanded && item.items && (
          <motion.div
            key="children"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ ...springs.smooth, opacity: { duration: 0.15 } }}
            className="overflow-hidden"
          >
            <div className="mt-0.5 space-y-0.5 border-l border-lucid-border ml-3 pl-0">
              {item.items.map((child) => (
                <SidebarGroup key={child.href ?? child.title} item={child} depth={depth + 1} />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

interface SidebarProps {
  items: SidebarItem[];
  className?: string;
}

export function Sidebar({ items, className }: SidebarProps) {
  return (
    <aside
      className={cn(
        "hidden lg:flex flex-col w-64 shrink-0",
        className,
      )}
      aria-label="Documentation navigation"
    >
      <ScrollArea className="flex-1 pt-8 pb-6 pr-4">
        <nav className="space-y-0.5">
          {items.map((item) => (
            <SidebarGroup key={item.href ?? item.title} item={item} />
          ))}
        </nav>
      </ScrollArea>
    </aside>
  );
}
