import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/** Compact human-readable count: ``61_100_840`` → ``"61.1M"``,
 *  ``1_400_000_000`` → ``"1.4B"``, ``234_000`` → ``"234K"``.
 *  One decimal of precision below 100; integer above.  Used by the
 *  model-factory cards to render ``param_count`` as a short tag. */
export function formatCompactCount(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "";
  const tiers: Array<[number, string]> = [
    [1_000_000_000, "B"],
    [1_000_000,     "M"],
    [1_000,         "K"],
  ];
  for (const [unit, suffix] of tiers) {
    if (n >= unit) {
      const v = n / unit;
      const formatted = v >= 100
        ? Math.round(v).toString()
        : v.toFixed(1).replace(/\.0$/, "");
      return `${formatted}${suffix}`;
    }
  }
  return n.toString();
}
