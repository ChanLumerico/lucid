/**
 * localStorage-backed recent-pages tracker used by the Cmd+K search
 * dialog's empty state.  Append-on-visit, dedup-by-href, cap at the
 * latest N entries so we don't grow unbounded across a long session.
 *
 * Storage shape:
 *
 *   key   = ``"lucid-docs-recent-pages"``
 *   value = JSON ``RecentPage[]`` — newest first.
 *
 * All accesses defend against ``localStorage`` being unavailable
 * (private-window restrictions, storage quota, manual disable) so the
 * search dialog still works when persistence fails.
 */

export interface RecentPage {
  href: string;
  title: string;
  /** UNIX ms timestamp.  Currently unused by the renderer but kept
   *  available for future "last 24h" / relative-time displays. */
  visitedAt: number;
}

const STORAGE_KEY = "lucid-docs-recent-pages";
const MAX_RECENTS = 8;

/** Read the recent-pages list from localStorage.  Returns ``[]`` on
 *  storage errors, parse failures, or when nothing has been recorded
 *  yet.  Safe to call from any client component — guards against the
 *  initial server-render pass where ``window`` is undefined. */
export function getRecentPages(): RecentPage[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (p): p is RecentPage =>
        typeof p === "object"
        && p !== null
        && typeof p.href === "string"
        && typeof p.title === "string"
        && typeof p.visitedAt === "number",
    );
  } catch {
    return [];
  }
}

/** Detect a quota-exceeded error.  Browser-specific codes diverge —
 *  Safari throws ``QuotaExceededError`` with name; Chrome / Firefox
 *  use the DOMException name; old IE used code 22.  Cover all three
 *  so the retry path triggers on any browser. */
function _isQuotaExceeded(e: unknown): boolean {
  if (!(e instanceof Error)) return false;
  const name = e.name;
  const code = (e as { code?: number }).code;
  return (
    name === "QuotaExceededError"
    || name === "NS_ERROR_DOM_QUOTA_REACHED"
    || code === 22
    || code === 1014
  );
}

/** Append a page visit to the head of the recents list.  Dedup-by-href
 *  so revisiting a page promotes it instead of duplicating; cap at
 *  ``MAX_RECENTS`` so the list stays scannable in the dialog. */
export function addRecentPage(page: Omit<RecentPage, "visitedAt">): void {
  if (typeof window === "undefined") return;
  if (!page.href || !page.title) return;
  const current = getRecentPages();
  // Drop any prior entry for this href so the new visit takes its
  // place at the front of the list.  Comparing href is enough —
  // hash fragments are stripped at the caller so different anchor
  // links to the same page count as one entry.
  const deduped = current.filter((p) => p.href !== page.href);
  const next: RecentPage[] = [
    { href: page.href, title: page.title, visitedAt: Date.now() },
    ...deduped,
  ].slice(0, MAX_RECENTS);
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch (e) {
    if (_isQuotaExceeded(e)) {
      // Storage full — drop our key entirely and retry with just
      // the new entry.  Better to lose old recents than fail to
      // record the current visit.
      try {
        window.localStorage.removeItem(STORAGE_KEY);
        window.localStorage.setItem(
          STORAGE_KEY,
          JSON.stringify([{ href: page.href, title: page.title, visitedAt: Date.now() }]),
        );
      } catch {
        // Still failing — give up silently; this is best-effort.
      }
    }
    // Storage disabled / other write failure — accept that recents
    // won't survive the session.
  }
}

/** Clear the recents list.  Exposed for a future "Clear history"
 *  control inside the dialog. */
export function clearRecentPages(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Ignore.
  }
}
