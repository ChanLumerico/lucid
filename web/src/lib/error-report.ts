/**
 * GitHub issue link builder used by every error-boundary surface.
 * Pre-fills the title + body so a user who clicks "Report this" lands
 * on the new-issue form already containing the page URL, the error
 * digest (when Next gives us one), and the user agent — the minimal
 * triage info we'd ask for anyway.
 *
 * No telemetry — the docs site doesn't ship a network reporter.
 * Surfacing a one-click GitHub link is the lightest hook that
 * actually closes the loop with users who hit a real error.
 */

const REPO = "ChanLumerico/lucid";

interface BuildIssueArgs {
  title: string;
  error: Error & { digest?: string };
  /** Page section the error happened in — used in the issue body so
   *  triage can route faster.  ``"/api"`` / ``"/docs"`` / ``"/"``. */
  section?: string;
}

export function buildIssueUrl({ title, error, section }: BuildIssueArgs): string {
  // ``window`` is guaranteed available — every caller is inside a
  // ``"use client"`` boundary.  Fall back to a safe default if the
  // tree somehow renders this on the server.
  const url = typeof window !== "undefined" ? window.location.href : "(unknown)";
  const ua = typeof navigator !== "undefined" ? navigator.userAgent : "(unknown)";

  const bodyLines = [
    "### Page URL",
    url,
    "",
    "### Section",
    section ?? "(unknown)",
    "",
    "### Error message",
    "```",
    error.message || "(no message)",
    "```",
  ];
  if (error.digest) {
    bodyLines.push(
      "",
      "### Next.js digest",
      "```",
      error.digest,
      "```",
    );
  }
  bodyLines.push(
    "",
    "### Browser",
    "```",
    ua,
    "```",
    "",
    "### What were you doing?",
    "<!-- Replace this with what triggered the error. -->",
  );

  const params = new URLSearchParams({
    title,
    body: bodyLines.join("\n"),
    labels: "docs,bug",
  });
  return `https://github.com/${REPO}/issues/new?${params.toString()}`;
}
