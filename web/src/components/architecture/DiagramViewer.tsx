import { cn } from "@/lib/utils";

/**
 * Same-origin iframe around an archify viewer, with no frame of its own: the
 * viewer opens in its stage layout, and ``public/archify/host.js`` lays the
 * diagram straight onto the page — no canvas box, no toolbar — repaints it
 * with the site's tokens and holds it to the site's theme.  Guided chapters,
 * finder, focus, route probe, lens and zoom stay live.
 */
export function DiagramViewer({
  src,
  title,
  className,
}: {
  src: string;
  title: string;
  className?: string;
}) {
  return (
    <div className={cn("relative", className)}>
      {/* Sits under the (transparent until painted) iframe while it loads. */}
      <div
        aria-hidden
        className="absolute inset-0 flex items-center justify-center text-xs text-lucid-text-low"
      >
        Loading diagram…
      </div>
      <iframe
        src={src}
        title={title}
        allow="clipboard-read; clipboard-write"
        className="relative block h-full w-full border-0"
      />
    </div>
  );
}
