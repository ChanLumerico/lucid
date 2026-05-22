"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { Dialog, DialogContent, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

interface Shortcut {
  /** Display keys.  Each string is rendered as a separate ``<kbd>``;
   *  multi-key sequences (``"g h"``) are split on whitespace and
   *  rendered as a row of separate keys. */
  keys: string[];
  label: string;
  /** Logical group used to bucket shortcuts in the help overlay so
   *  related rows sit together. */
  group: "Global" | "Navigation" | "View";
  /** Runtime handler — invoked by the global ``keydown`` dispatcher
   *  when the user presses the matching sequence.  ``ctx`` exposes the
   *  router + the function that toggles each modal so handlers don't
   *  need to import from anywhere themselves. */
  run: (ctx: ShortcutCtx) => void;
}

interface ShortcutCtx {
  router: ReturnType<typeof useRouter>;
  toggleHelp: () => void;
  toggleSearch: () => void;
}

const SHORTCUTS: Shortcut[] = [
  // ─── Global ────────────────────────────────────────────────────────────
  {
    keys: ["⌘", "K"],
    label: "Open search",
    group: "Global",
    run: ({ toggleSearch }) => toggleSearch(),
  },
  {
    keys: ["?"],
    label: "Show this help",
    group: "Global",
    run: ({ toggleHelp }) => toggleHelp(),
  },
  // ─── Navigation ────────────────────────────────────────────────────────
  {
    keys: ["g", "h"],
    label: "Go to home",
    group: "Navigation",
    run: ({ router }) => router.push("/"),
  },
  {
    keys: ["g", "a"],
    label: "Go to API reference",
    group: "Navigation",
    run: ({ router }) => router.push("/api"),
  },
  {
    keys: ["g", "d"],
    label: "Go to docs / guides",
    group: "Navigation",
    run: ({ router }) => router.push("/docs"),
  },
  {
    keys: ["g", "c"],
    label: "Go to changelog",
    group: "Navigation",
    run: ({ router }) => router.push("/changelog"),
  },
  // ─── View ──────────────────────────────────────────────────────────────
  {
    keys: ["t"],
    label: "Toggle theme",
    group: "View",
    run: () => {
      // Dispatch a click on the existing ``ThemeToggle`` button —
      // single source of truth for the toggle behaviour stays in
      // that component rather than duplicating localStorage logic.
      const btn = document.querySelector<HTMLButtonElement>(
        'button[aria-label*="Switch to"]',
      );
      btn?.click();
    },
  },
];

const GROUPS: Shortcut["group"][] = ["Global", "Navigation", "View"];

interface KeyboardShortcutsProps {
  /** Click-to-open trigger uses this; ``Header`` toggles via state
   *  so the ``?`` global keypress and any UI affordance share the
   *  same modal state. */
  open: boolean;
  onClose: () => void;
  onOpen: () => void;
}

/** Global keyboard-shortcuts modal + dispatcher.  Mounted once at the
 *  layout root so the keydown listener catches any focus context that
 *  isn't an editable surface.
 *
 *  Multi-key sequences (``g h``, ``g a``) use a tiny "pending prefix"
 *  state — when the user presses ``g``, we wait up to 1500 ms for the
 *  second key; if nothing follows, the sequence times out and the
 *  next key starts fresh.  This matches the GitHub / Linear convention. */
export function KeyboardShortcuts({ open, onClose, onOpen }: KeyboardShortcutsProps) {
  const router = useRouter();
  const pendingRef = React.useRef<string | null>(null);
  const pendingTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  // ``toggleSearch`` is a no-op stub here — the SearchDialog is opened
  // via its own ``⌘K`` listener inside ``Header``.  We keep the slot
  // in the ctx object so future shortcuts can drive the search modal
  // through this same dispatcher without re-plumbing.
  const ctx = React.useMemo<ShortcutCtx>(
    () => ({
      router,
      toggleHelp: () => (open ? onClose() : onOpen()),
      toggleSearch: () => {
        // Dispatch ``⌘+K`` as a synthetic event so the existing Header
        // listener handles it.  Avoids exposing SearchDialog state up
        // through the layout tree.
        const ev = new KeyboardEvent("keydown", {
          key: "k",
          metaKey: true,
          bubbles: true,
        });
        window.dispatchEvent(ev);
      },
    }),
    [router, open, onOpen, onClose],
  );

  React.useEffect(() => {
    function isEditableTarget(el: EventTarget | null): boolean {
      if (!(el instanceof HTMLElement)) return false;
      const tag = el.tagName;
      return (
        tag === "INPUT"
        || tag === "TEXTAREA"
        || tag === "SELECT"
        || el.isContentEditable
      );
    }

    function clearPending() {
      pendingRef.current = null;
      if (pendingTimerRef.current) {
        clearTimeout(pendingTimerRef.current);
        pendingTimerRef.current = null;
      }
    }

    function onKeyDown(e: KeyboardEvent) {
      // Don't intercept while typing in a form — power-user shortcuts
      // shouldn't fight with text entry.
      if (isEditableTarget(e.target)) return;
      // Modifier-locked global shortcuts have their own handlers
      // (``⌘K`` lives in Header) — skip here.
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      const key = e.key;

      // ``?`` works without shift on some layouts; both branches OK.
      if (key === "?" || (e.shiftKey && key === "/")) {
        e.preventDefault();
        clearPending();
        ctx.toggleHelp();
        return;
      }

      // Single-key shortcuts.
      const singleHit = SHORTCUTS.find(
        (s) => s.keys.length === 1 && s.keys[0].toLowerCase() === key.toLowerCase(),
      );
      if (singleHit) {
        e.preventDefault();
        clearPending();
        singleHit.run(ctx);
        return;
      }

      // Multi-key sequences (``g h`` / ``g a`` …).  If there's a
      // pending prefix, try the full match; otherwise stash this key
      // as a potential prefix.
      const lower = key.toLowerCase();
      if (pendingRef.current !== null) {
        const seq = [pendingRef.current, lower];
        const hit = SHORTCUTS.find(
          (s) =>
            s.keys.length === 2
            && s.keys[0].toLowerCase() === seq[0]
            && s.keys[1].toLowerCase() === seq[1],
        );
        clearPending();
        if (hit) {
          e.preventDefault();
          hit.run(ctx);
        }
        return;
      }
      // Start a new prefix when this key is the first of any 2-key seq.
      const prefixOfAny = SHORTCUTS.some(
        (s) => s.keys.length === 2 && s.keys[0].toLowerCase() === lower,
      );
      if (prefixOfAny) {
        pendingRef.current = lower;
        pendingTimerRef.current = setTimeout(clearPending, 1500);
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      clearPending();
    };
  }, [ctx]);

  return (
    <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
      <DialogContent>
        <DialogTitle className="px-5 pt-5 pb-1 text-sm font-semibold text-lucid-text-high">
          Keyboard shortcuts
        </DialogTitle>
        <DialogDescription className="sr-only">
          Press the listed key combinations from anywhere on the docs
          site (except inside editable fields).  Multi-key sequences
          (e.g. ``g h``) accept up to 1.5 s between keys.
        </DialogDescription>
        <div className="max-h-[60vh] overflow-y-auto px-5 pb-5 pt-2">
          {GROUPS.map((group) => {
            const rows = SHORTCUTS.filter((s) => s.group === group);
            if (rows.length === 0) return null;
            return (
              <section key={group} className="mb-4 last:mb-0">
                <p className="mb-1.5 text-[10px] font-semibold tracking-widest uppercase text-lucid-text-disabled">
                  {group}
                </p>
                <ul className="space-y-1">
                  {rows.map((row, i) => (
                    <li
                      key={i}
                      className={cn(
                        "flex items-center justify-between gap-3 rounded-lg",
                        "px-2 py-1.5 hover:bg-lucid-elevated transition-colors",
                      )}
                    >
                      <span className="text-sm text-lucid-text-mid">{row.label}</span>
                      <span className="flex items-center gap-1">
                        {row.keys.map((k, ki) => (
                          <kbd
                            key={ki}
                            className={cn(
                              "inline-flex h-6 min-w-6 items-center justify-center rounded",
                              "border border-lucid-border bg-lucid-surface px-1.5",
                              "font-mono text-[11px] text-lucid-text-mid",
                            )}
                          >
                            {k}
                          </kbd>
                        ))}
                      </span>
                    </li>
                  ))}
                </ul>
              </section>
            );
          })}
          <p className="mt-2 text-[11px] text-lucid-text-disabled">
            Shortcuts are inactive while typing in inputs / textareas.
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
}
