/**
 * Convert a GitHub ``/blob/<sha>/...#L42`` source URL emitted by
 * ``web/scripts/build-api-data.py`` into the matching ``/edit/<branch>/...``
 * URL.  GitHub's edit-mode endpoint takes a *branch* name in place of
 * the SHA — opening ``/edit/<sha>/...`` works visually but blocks the
 * commit step because you can't push to a SHA.  Using ``main`` lets
 * the editor offer "Propose changes" + auto-fork-and-PR.
 *
 * Inputs that aren't in the expected ``blob/<sha>`` shape are returned
 * as-is; the caller decides whether to render the link based on
 * whether the input changed.
 */

const _BRANCH = "main";

export function sourceToEditUrl(sourceUrl: string | null | undefined): string | null {
  if (!sourceUrl) return null;
  // Match ``https://github.com/<owner>/<repo>/blob/<rev>/<path>...``
  // where ``<rev>`` is the SHA we want to swap for the branch name.
  // Trailing ``#Lnnn`` anchors are stripped — GitHub's edit page
  // doesn't honour them, and they'd land readers in an unexpected
  // sub-section of the editor.
  const match = sourceUrl.match(
    /^(https:\/\/github\.com\/[^/]+\/[^/]+)\/blob\/[^/]+\/(.+?)(?:#L\d+(?:-L\d+)?)?$/,
  );
  if (!match) return null;
  const [, repo, path] = match;
  return `${repo}/edit/${_BRANCH}/${path}`;
}
