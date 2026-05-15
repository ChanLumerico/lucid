"use client";

/**
 * ApiKindBadge — visual kind indicators for API reference entries.
 *
 * Color source of truth: globals.css @theme block (--color-api-*)
 * To retheme, edit globals.css only — nothing here needs to change.
 */

import type { ApiLabel, ApiClassKind } from "@/lib/types";
import { cn } from "@/lib/utils";
import {
  getMemberNameColor as _getMemberNameColor,
  getClassNameColor as _getClassNameColor,
  getClassHoverBorder as _getClassHoverBorder,
} from "@/lib/api-kind-utils";

// Re-export so existing imports from this file still work
export { _getMemberNameColor as getMemberNameColor, _getClassNameColor as getClassNameColor, _getClassHoverBorder as getClassHoverBorder };

// ---------------------------------------------------------------------------
// Pill primitive
// ---------------------------------------------------------------------------

function Pill({ label, className }: { label: string; className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex shrink-0 select-none items-center",
        "rounded px-1.5 py-0.5",
        "text-[9px] font-bold tracking-widest uppercase",
        "border",
        className,
      )}
    >
      {label}
    </span>
  );
}

// Color helpers are defined in @/lib/api-kind-utils (server-safe, no "use client")
// and re-exported above from this file for backwards compatibility.

// ---------------------------------------------------------------------------
// Named kind badges
// ---------------------------------------------------------------------------

/** Concrete class */
export function ClassBadge({
  kind = "regular",
  className,
}: {
  kind?: ApiClassKind;
  className?: string;
}) {
  if (kind === "abstract") {
    return (
      <Pill
        label="abstract"
        className={cn(
          "border-dashed bg-api-class-abstract/10 border-api-class-abstract/35 text-api-class-abstract",
          className,
        )}
      />
    );
  }
  if (kind === "dataclass") {
    return (
      <Pill
        label="data"
        className={cn(
          "bg-api-class-dataclass/10 border-api-class-dataclass/35 text-api-class-dataclass",
          className,
        )}
      />
    );
  }
  return (
    <Pill
      label="class"
      className={cn(
        "bg-api-class/10 border-api-class/30 text-api-class",
        className,
      )}
    />
  );
}

/** Regular function / method */
export function FnBadge({ className }: { className?: string }) {
  return (
    <Pill
      label="fn"
      className={cn("bg-api-fn/10 border-api-fn/30 text-api-fn", className)}
    />
  );
}

/** @property */
export function PropBadge({
  writable = false,
  className,
}: {
  writable?: boolean;
  className?: string;
}) {
  return (
    <Pill
      label={writable ? "prop rw" : "prop"}
      className={cn(
        "bg-api-prop/10 border-api-prop/30 text-api-prop",
        className,
      )}
    />
  );
}

/** @classmethod */
export function ClsMethodBadge({ className }: { className?: string }) {
  return (
    <Pill
      label="cls"
      className={cn("bg-api-cls/10 border-api-cls/30 text-api-cls", className)}
    />
  );
}

/** @staticmethod */
export function StaticBadge({ className }: { className?: string }) {
  return (
    <Pill
      label="static"
      className={cn(
        "bg-api-static/10 border-api-static/30 text-api-static",
        className,
      )}
    />
  );
}

/** @abstractmethod */
export function AbstractBadge({ className }: { className?: string }) {
  return (
    <Pill
      label="abstract"
      className={cn(
        "border-dashed bg-api-abstract/8 border-api-abstract/30 text-api-abstract",
        className,
      )}
    />
  );
}

/** __dunder__ methods */
export function DunderBadge({ className }: { className?: string }) {
  return (
    <Pill
      label="dunder"
      className={cn(
        "bg-api-dunder/10 border-api-dunder/30 text-api-dunder",
        className,
      )}
    />
  );
}

// ---------------------------------------------------------------------------
// AutoKindBadge — picks the correct badge automatically
// Priority: property > classmethod > staticmethod > abstractmethod > dunder > fn
// ---------------------------------------------------------------------------

interface AutoKindBadgeProps {
  labels: ApiLabel[];
  /** Name is used to detect __dunder__ methods */
  name?: string;
  fallback?: "fn" | "class";
  classKind?: ApiClassKind;
  className?: string;
}

export function AutoKindBadge({
  labels,
  name,
  fallback = "fn",
  classKind = "regular",
  className,
}: AutoKindBadgeProps) {
  const isWritable = labels.includes("writable");

  if (labels.includes("property"))       return <PropBadge writable={isWritable} className={className} />;
  if (labels.includes("classmethod"))    return <ClsMethodBadge className={className} />;
  if (labels.includes("staticmethod"))   return <StaticBadge className={className} />;
  if (labels.includes("abstractmethod")) return <AbstractBadge className={className} />;

  if (name && name.startsWith("__") && name.endsWith("__")) {
    return <DunderBadge className={className} />;
  }

  if (fallback === "class") return <ClassBadge kind={classKind} className={className} />;
  return <FnBadge className={className} />;
}
