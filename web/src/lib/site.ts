// Where the docs are served and where the code lives.  Every absolute URL
// the site emits (metadata, sitemap, robots, GitHub links) is built from
// these constants, so a move of the repository or of the Pages host is a
// one-file change.

/** GitHub Pages serves the site under ``/<repo>/``.  Mirrors ``basePath`` in
 *  next.config.ts — prod only, so ``next dev`` stays at plain ``/``. */
export const BASE_PATH = process.env.NODE_ENV === "production" ? "/lucid" : "";

/** Scheme + host of the deployed site, without the base path.  The env
 *  override is read at build time only (server code), so it never needs a
 *  ``NEXT_PUBLIC_`` twin to be inlined into client bundles. */
export const SITE_ORIGIN =
  process.env.LUCID_DOCS_ORIGIN ??
  process.env.NEXT_PUBLIC_LUCID_DOCS_ORIGIN ??
  "https://lucid-official.github.io";

/** Absolute URL of the site root, base path included and no trailing slash —
 *  ``${SITE_URL}/api/lucid.nn`` is a real page. */
export const SITE_URL = `${SITE_ORIGIN}${BASE_PATH}`;

/** URL of a brand-kit file, by its base name in ``assets/`` — the copies
 *  ``scripts/sync-brand.mjs`` puts in ``public/brand/``.  Base path included:
 *  every consumer (``<img>``, metadata ``icons``, the web manifest) emits it
 *  as written, and Next adds the base path to none of them. */
export function brandAsset(file: string): string {
  return `${BASE_PATH}/brand/${file}`;
}

/** ``alternates`` metadata naming the page at ``path`` (base-relative, with a
 *  leading slash) as its own canonical URL.  Spelled out absolute because
 *  Next joins a relative canonical onto ``metadataBase`` without the base
 *  path — and ``metadataBase`` stays origin-only, since the image URLs
 *  (``brandAsset("og-image.png")``) already carry it. */
export function canonical(path: string): { canonical: string } {
  return { canonical: `${SITE_URL}${path}` };
}

/** ``owner/name`` of the GitHub repository. */
export const REPO = "lucid-official/lucid";

export const REPO_URL = `https://github.com/${REPO}`;
