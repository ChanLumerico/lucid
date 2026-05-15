import { createHighlighter, type Highlighter } from "shiki";

let highlighterPromise: Promise<Highlighter> | null = null;

function getHighlighter(): Promise<Highlighter> {
  if (!highlighterPromise) {
    highlighterPromise = createHighlighter({
      themes: ["one-dark-pro", "github-light"],
      langs: ["python", "bash", "typescript", "json"],
    });
  }
  return highlighterPromise;
}

export async function highlight(code: string, lang: string = "python"): Promise<string> {
  const hl = await getHighlighter();
  return hl.codeToHtml(code, {
    lang,
    theme: "one-dark-pro",
  });
}

export async function highlightSignature(sig: string): Promise<string> {
  return highlight(sig, "python");
}
