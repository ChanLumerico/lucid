"use client";

import * as React from "react";

/** Document-level click delegate that copies the text content of any
 *  *inline* ``<code>`` element on click.  Mounted once at the layout
 *  root so any page — server-rendered MathText, MDX guides, etc. —
 *  picks up the behaviour without per-component wiring.
 *
 *  Filter rules:
 *    * ``<code>`` inside ``<pre>`` is skipped — code blocks already
 *      have their own ``CopyButton`` overlay.
 *    * ``<code>`` inside an interactive ancestor (``<a>`` /
 *      ``<button>``) is skipped — clicking the link should navigate,
 *      not silently copy a fragment.
 *    * Empty / whitespace-only text is skipped.
 *
 *  Feedback: we briefly stamp ``data-copied="1"`` on the element.  The
 *  matching CSS rule in ``globals.css`` swaps the foreground colour
 *  to ``--color-lucid-success`` for ~1 second so the user sees a
 *  visual confirmation without us mutating React state for every
 *  click. */
export function InlineCodeCopy() {
  React.useEffect(() => {
    function isInteractive(el: Element | null): boolean {
      while (el) {
        if (
          el instanceof HTMLAnchorElement
          || el instanceof HTMLButtonElement
        ) {
          return true;
        }
        el = el.parentElement;
      }
      return false;
    }

    function onClick(e: MouseEvent) {
      const target = e.target;
      if (!(target instanceof Element)) return;
      // ``closest`` finds the nearest ``<code>`` ancestor including
      // the target itself; we filter out the ``<pre>``-wrapped case
      // afterwards.  This lets ``<span>``-styled tokens inside a
      // code element still trigger the copy.
      const code = target.closest("code");
      if (!code) return;
      // Skip block-level code (inside ``<pre>``) — those have their
      // own copy button.
      if (code.closest("pre")) return;
      // Skip interactive ancestors so we don't hijack link clicks.
      if (isInteractive(code.parentElement)) return;

      const text = code.textContent?.trim();
      if (!text) return;

      // Suppress text selection on a single rapid double-click —
      // copying short identifiers shouldn't simultaneously trigger
      // text selection that the user has to clear.
      e.preventDefault();
      const sel = window.getSelection();
      sel?.removeAllRanges();

      navigator.clipboard
        .writeText(text)
        .then(() => {
          code.setAttribute("data-copied", "1");
          setTimeout(() => code.removeAttribute("data-copied"), 1000);
        })
        .catch(() => {
          // Clipboard blocked (insecure context, restricted) — no-op.
          // We don't fall back to ``execCommand("copy")`` because that
          // path is deprecated and the failure mode here is benign.
        });
    }

    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, []);

  return null;
}
