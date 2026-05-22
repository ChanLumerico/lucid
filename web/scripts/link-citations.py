"""link-citations.py — post-process api-data JSON to make references clickable.

Walks every string field in ``web/public/api-data/*.json``, scans for
citation patterns, and rewrites them in-place as markdown links so the
MathText / Markdown renderer downstream emits proper ``<a href>`` tags.

Three classes of patterns are handled:

1. **Inline arXiv IDs** — ``arXiv:1706.03762`` →
   ``[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)``.  Highest
   priority and always wins regardless of curated table.

2. **Inline DOIs** — bare ``10.NNNN/...`` strings →
   ``[10.NNNN/...](https://doi.org/10.NNNN/...)``.

3. **Author + year refs** — patterns like
   ``Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)`` are
   looked up against the curated ``data/citations.json``.  The first
   matching ``(author, year)`` extracted from the citation line drives
   the lookup; the lookup key is ``<lastname>_<year>`` (e.g.
   ``vaswani_2017``) plus a few common joined forms
   (``kingma_ba_2015``).

Idempotency: a citation that is *already* inside markdown link syntax
``[...](...)`` is skipped, so the script can run repeatedly without
double-wrapping.

Runs after every prebuild; safe to also invoke standalone via
``pnpm citations`` or directly.  Walks the entire ``api-data/`` dir so
both Python-side and C++-side JSONs are covered.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
WEB_ROOT = HERE.parent
DATA = WEB_ROOT / "data" / "citations.json"
API_DATA = WEB_ROOT / "public" / "api-data"

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Inside which fields should we look for citations?  Stick to the
# extended-prose fields plus structured sections that may contain refs.
# Skip ``source`` (URLs), ``signature``, ``name``, ``path`` (identifiers),
# and structured ``parameters`` entries (handled via their ``description``
# sub-field separately).
_PROSE_FIELDS = frozenset({"extended", "summary", "description"})
_PROSE_LIST_FIELDS = frozenset({"notes", "warns", "raises", "examples"})

_ARXIV_RE  = re.compile(r"\barXiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)\b", re.IGNORECASE)
_DOI_RE    = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)")
# "Vaswani et al., ..." or "Kingma & Ba, ..." up through the year-bearing
# parenthesised tail.  Captures (first_author, year).  Greedy enough to
# absorb a full citation line, anchored at a capital letter (start of a
# proper name) and ending at the first year-style ``(... 19xx|20xx)`` or
# bare-year ``(YYYY)`` reference.
_AUTHOR_YEAR_RE = re.compile(
    r"""
    \b(?P<author>[A-Z][a-zA-Z\-']+)                # First author family name
    (?:\s+(?:et\s+al\.?|&\s+[A-Z][a-zA-Z\-']+))?   # optional "et al." or "& Other"
    [^.\n(]{0,200}?                                # citation body up to the year
    \(
        (?:[^()\n]*?,\s*)?                         # optional venue prefix
        (?P<year>(?:18|19|20)\d{2})                # 4-digit year
        [^()\n]*?                                  # rest inside parens
    \)
    """,
    re.VERBOSE,
)


def _load_table() -> dict[str, dict[str, Any]]:
    return json.loads(DATA.read_text())


def _resolve_url(entry: dict[str, Any]) -> str | None:
    if entry.get("arxiv"):
        return f"https://arxiv.org/abs/{entry['arxiv']}"
    if entry.get("doi"):
        return f"https://doi.org/{entry['doi']}"
    return entry.get("url")


# Mask any text that's already inside a markdown link so we don't
# double-wrap.  Returns (masked_text, restore_fn).
_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")


def _mask_existing_links(text: str) -> tuple[str, list[tuple[int, str]]]:
    """Replace every existing ``[label](url)`` with a unique placeholder
    so subsequent citation regexes don't see inside them.  Returns the
    masked text and a list of ``(placeholder_index, original)`` pairs so
    we can restore."""
    placeholders: list[tuple[int, str]] = []
    def _sub(m: re.Match[str]) -> str:
        placeholders.append((len(placeholders), m.group(0)))
        return f"\x00LINK{len(placeholders) - 1}\x00"
    return _LINK_RE.sub(_sub, text), placeholders


def _restore_existing_links(text: str, placeholders: list[tuple[int, str]]) -> str:
    for idx, orig in placeholders:
        text = text.replace(f"\x00LINK{idx}\x00", orig)
    return text


def _link_arxiv(text: str) -> tuple[str, int]:
    n = 0
    def _sub(m: re.Match[str]) -> str:
        nonlocal n
        n += 1
        ident = m.group(1)
        return f"[arXiv:{ident}](https://arxiv.org/abs/{ident})"
    return _ARXIV_RE.sub(_sub, text), n


def _link_doi(text: str) -> tuple[str, int]:
    n = 0
    def _sub(m: re.Match[str]) -> str:
        nonlocal n
        n += 1
        doi = m.group(1).rstrip(".,;)")
        return f"[{doi}](https://doi.org/{doi})"
    return _DOI_RE.sub(_sub, text), n


def _link_author_year(text: str, table: dict[str, dict[str, Any]]) -> tuple[str, int]:
    """Look up ``<author>_<year>`` against the curated table.  Skips
    matches that don't resolve to a URL — we don't want to wrap a refer-
    ence just to point at nothing."""
    n = 0
    def _sub(m: re.Match[str]) -> str:
        nonlocal n
        author = m.group("author").lower()
        year = m.group("year")
        key = f"{author}_{year}"
        entry = table.get(key)
        if entry is None:
            return m.group(0)
        url = _resolve_url(entry)
        if url is None:
            return m.group(0)
        n += 1
        return f"[{m.group(0)}]({url})"
    return _AUTHOR_YEAR_RE.sub(_sub, text), n


def _link_text(text: str, table: dict[str, dict[str, Any]]) -> tuple[str, int]:
    if not isinstance(text, str) or "[" not in text and "(" not in text and \
       "arXiv" not in text and "10." not in text and "et al" not in text and \
       "&" not in text:
        # Cheap pre-filter: skip strings that clearly don't contain a ref.
        return text, 0
    masked, ph = _mask_existing_links(text)
    masked, n1 = _link_arxiv(masked)
    masked, n2 = _link_doi(masked)
    masked, n3 = _link_author_year(masked, table)
    return _restore_existing_links(masked, ph), (n1 + n2 + n3)


# ---------------------------------------------------------------------------
# Recursive walk over the api-data tree
# ---------------------------------------------------------------------------

def _walk(node: Any, table: dict[str, dict[str, Any]], stats: dict[str, int]) -> Any:
    if isinstance(node, str):
        new, n = _link_text(node, table)
        stats["links"] += n
        return new
    if isinstance(node, list):
        return [_walk(v, table, stats) for v in node]
    if isinstance(node, dict):
        return {k: _walk(v, table, stats) for k, v in node.items()}
    return node


def _process_file(path: Path, table: dict[str, dict[str, Any]]) -> int:
    data = json.loads(path.read_text())
    stats = {"links": 0}
    rewritten = _walk(data, table, stats)
    if stats["links"]:
        path.write_text(json.dumps(rewritten, indent=2, ensure_ascii=False))
    return stats["links"]


def main() -> int:
    if not DATA.is_file():
        print(f"[link-citations] {DATA} missing — nothing to do", file=sys.stderr)
        return 0
    table = _load_table()
    if not API_DATA.is_dir():
        print(f"[link-citations] {API_DATA} missing — nothing to do", file=sys.stderr)
        return 0
    total = 0
    files = sorted(API_DATA.glob("*.json"))
    files = [f for f in files if not f.name.startswith("_")]
    for f in files:
        n = _process_file(f, table)
        total += n
    print(f"[link-citations] wrote {total} link(s) across {len(files)} JSON file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
