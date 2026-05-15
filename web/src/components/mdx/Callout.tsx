import * as React from "react";
import { AlertCircle, Info, AlertTriangle, CheckCircle } from "lucide-react";
import { cn } from "@/lib/utils";

type CalloutVariant = "info" | "warning" | "danger" | "success" | "note";

const VARIANT_CONFIG: Record<
  CalloutVariant,
  { icon: React.ElementType; classes: string; iconClass: string }
> = {
  info: {
    icon: Info,
    classes:
      "border-lucid-blue/30 bg-lucid-blue/5 text-lucid-text-mid",
    iconClass: "text-lucid-blue",
  },
  note: {
    icon: Info,
    classes:
      "border-lucid-border bg-lucid-surface text-lucid-text-mid",
    iconClass: "text-lucid-text-low",
  },
  warning: {
    icon: AlertTriangle,
    classes:
      "border-lucid-warning/30 bg-lucid-warning/5 text-lucid-text-mid",
    iconClass: "text-lucid-warning",
  },
  danger: {
    icon: AlertCircle,
    classes:
      "border-lucid-error/30 bg-lucid-error/5 text-lucid-text-mid",
    iconClass: "text-lucid-error",
  },
  success: {
    icon: CheckCircle,
    classes:
      "border-lucid-success/30 bg-lucid-success/5 text-lucid-text-mid",
    iconClass: "text-lucid-success",
  },
};

interface CalloutProps {
  variant?: CalloutVariant;
  title?: string;
  children: React.ReactNode;
}

export function Callout({
  variant = "info",
  title,
  children,
}: CalloutProps) {
  const { icon: Icon, classes, iconClass } = VARIANT_CONFIG[variant];

  return (
    <div
      className={cn(
        "my-6 flex gap-3 rounded-xl border px-4 py-3.5 text-sm leading-relaxed",
        classes,
      )}
    >
      <Icon
        className={cn("mt-0.5 h-4 w-4 shrink-0", iconClass)}
        aria-hidden="true"
      />
      <div className="min-w-0">
        {title && (
          <p className="mb-1 font-semibold text-lucid-text-high">{title}</p>
        )}
        <div className="prose-callout">{children}</div>
      </div>
    </div>
  );
}
