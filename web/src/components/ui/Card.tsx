import type * as React from "react";
import Link from "next/link";
import { cn } from "@/lib/utils";

/**
 * Canonical clickable card — the single source of the card recipe used by
 * every overview / sub-package / family card.  Recipe (enforced by the
 * ``ui.card-shape`` audit contract):
 *
 *   rounded-xl · border border-lucid-border · bg-lucid-surface/40
 *   · transition-colors · hover:bg-lucid-surface hover:border-lucid-primary/40
 *
 * Pass ``className`` only for layout/padding additions — never to override the
 * shape/border/hover tokens (that would be design drift the audit rejects).
 */
export function Card({
  href,
  className,
  children,
}: {
  href: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <Link
      href={href}
      className={cn(
        "group block rounded-xl border border-lucid-border bg-lucid-surface/40",
        "transition-colors hover:bg-lucid-surface hover:border-lucid-primary/40",
        className,
      )}
    >
      {children}
    </Link>
  );
}
