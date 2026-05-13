"use client";

import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  [
    "inline-flex items-center justify-center gap-2 whitespace-nowrap",
    "rounded-lg text-sm font-medium transition-all duration-150",
    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-lucid-primary focus-visible:ring-offset-2 focus-visible:ring-offset-lucid-bg",
    "disabled:pointer-events-none disabled:opacity-40",
    "select-none cursor-pointer",
  ],
  {
    variants: {
      variant: {
        default: [
          "bg-lucid-primary text-white shadow-sm",
          "hover:bg-lucid-primary-dark active:scale-[0.98]",
        ],
        secondary: [
          "bg-lucid-surface text-lucid-text-high border border-lucid-border",
          "hover:bg-lucid-elevated hover:border-lucid-border active:scale-[0.98]",
        ],
        ghost: [
          "text-lucid-text-mid",
          "hover:bg-lucid-surface hover:text-lucid-text-high",
        ],
        link: [
          "text-lucid-primary underline-offset-4",
          "hover:underline hover:text-lucid-primary-light",
          "p-0 h-auto",
        ],
        destructive: [
          "bg-lucid-error/15 text-lucid-error border border-lucid-error/30",
          "hover:bg-lucid-error/25 active:scale-[0.98]",
        ],
        outline: [
          "border border-lucid-border text-lucid-text-mid bg-transparent",
          "hover:bg-lucid-surface hover:text-lucid-text-high hover:border-lucid-primary/40",
          "active:scale-[0.98]",
        ],
      },
      size: {
        sm: "h-8 px-3 text-xs rounded-md",
        default: "h-9 px-4",
        lg: "h-11 px-6 text-base rounded-xl",
        icon: "h-9 w-9",
        "icon-sm": "h-7 w-7 rounded-md",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        ref={ref}
        className={cn(buttonVariants({ variant, size }), className)}
        {...props}
      />
    );
  },
);
Button.displayName = "Button";

export { Button, buttonVariants };
