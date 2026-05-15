import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-semibold tracking-wide uppercase transition-colors",
  {
    variants: {
      variant: {
        default:
          "bg-lucid-primary/15 text-lucid-primary border border-lucid-primary/25",
        secondary:
          "bg-lucid-surface text-lucid-text-mid border border-lucid-border",
        success:
          "bg-lucid-success/15 text-lucid-success border border-lucid-success/25",
        warning:
          "bg-lucid-warning/15 text-lucid-warning border border-lucid-warning/25",
        error:
          "bg-lucid-error/15 text-lucid-error border border-lucid-error/25",
        blue:
          "bg-lucid-blue/15 text-lucid-blue border border-lucid-blue/25",
        outline:
          "border border-lucid-border text-lucid-text-mid bg-transparent",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <span className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
