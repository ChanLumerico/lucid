import { createHighlighter, type Highlighter } from "shiki";

let highlighterPromise: Promise<Highlighter> | null = null;

function getHighlighter(): Promise<Highlighter> {
  if (!highlighterPromise) {
    highlighterPromise = createHighlighter({
      // Two themes loaded so the same JSON build emits dual-token
      // styles — picked between at runtime via CSS variables.  This
      // keeps the static export small (no two HTML strings) and lets
      // the live theme toggle flip code blocks instantly with no
      // re-render.
      themes: ["one-dark-pro", "github-light"],
      langs: ["python", "bash", "typescript", "json"],
    });
  }
  return highlighterPromise;
}

/** Highlight ``code`` with both light + dark themes baked into the
 *  inline styles.  Shiki emits ``--shiki-light`` / ``--shiki-dark``
 *  custom properties on every token; CSS in ``globals.css`` swaps
 *  which one wins based on ``html[data-theme]``.  Default render is
 *  dark — the light variant is only applied under ``data-theme=light``. */
export async function highlight(code: string, lang: string = "python"): Promise<string> {
  const hl = await getHighlighter();
  return hl.codeToHtml(code, {
    lang,
    themes: {
      dark: "one-dark-pro",
      light: "github-light",
    },
    defaultColor: "dark",
  });
}

export async function highlightSignature(sig: string): Promise<string> {
  return highlight(sig, "python");
}
