// Skip-to-content keyboard affordance.  Visually hidden until focused
// (Tab from the address bar puts it first), then renders as an
// accessible button in the top-left.  WCAG 2.4.1 (Bypass Blocks).

export function SkipLink({ targetId = "main-content" }: { targetId?: string }) {
  return (
    <a
      href={`#${targetId}`}
      className={[
        "sr-only",
        "focus:not-sr-only",
        "focus:fixed focus:left-4 focus:top-3 focus:z-[60]",
        "focus:rounded-md focus:border focus:border-lucid-primary",
        "focus:bg-lucid-bg focus:px-3 focus:py-1.5",
        "focus:text-sm focus:font-medium focus:text-lucid-text-high",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-lucid-primary/60",
      ].join(" ")}
    >
      Skip to content
    </a>
  );
}
