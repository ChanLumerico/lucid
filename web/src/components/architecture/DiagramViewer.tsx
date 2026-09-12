"use client";

import * as React from "react";
import { useTheme } from "@/components/layout/ThemeProvider";
import { cn } from "@/lib/utils";

/** Posted by ``public/archify/host.js`` when the viewer's own theme toggle
 *  is used, so the page follows it.  The opposite direction needs no
 *  message: the bridge watches this document's ``data-theme`` directly. */
const THEME_MESSAGE = "lucid:archify-theme";

/**
 * Same-origin iframe around an archify viewer.  The full viewer stays live —
 * toolbar, guided chapters, finder, focus, route probe, export — while
 * ``host.js`` repaints it with the site's tokens and hides the title and fact
 * cards the page already renders natively.
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
  const frameRef = React.useRef<HTMLIFrameElement>(null);
  const { setTheme } = useTheme();

  React.useEffect(() => {
    function onMessage(e: MessageEvent) {
      if (e.origin !== window.location.origin) return;
      if (e.source !== frameRef.current?.contentWindow) return;
      const data = e.data as { type?: unknown; theme?: unknown } | null;
      if (data?.type !== THEME_MESSAGE) return;
      if (data.theme === "light" || data.theme === "dark") setTheme(data.theme);
    }
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, [setTheme]);

  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-xl border border-lucid-border bg-lucid-bg",
        className,
      )}
    >
      {/* Sits under the (transparent until painted) iframe while it loads. */}
      <div
        aria-hidden
        className="absolute inset-0 flex items-center justify-center text-xs text-lucid-text-low"
      >
        Loading diagram…
      </div>
      <iframe
        ref={frameRef}
        src={src}
        title={title}
        allow="fullscreen; clipboard-read; clipboard-write"
        allowFullScreen
        className="relative block h-full w-full border-0"
      />
    </div>
  );
}
