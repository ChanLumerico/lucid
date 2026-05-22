"""build-bibtex.py — emit ``public/citations.bib`` from the curated
``data/citations.json`` lookup.

The lookup is keyed by ``<lastname>_<year>`` and stores ``arxiv`` /
``doi`` / ``url`` / ``title``.  We synthesise a minimal BibTeX entry
per key — readers downloading the file can drop it into their reference
manager directly.

Aliases that share an arXiv ID (``kingma_2015`` and ``kingma_ba_2015``
both point at ``1412.6980``) collapse to a single entry — the first
key seen wins as the BibTeX cite-key, the rest emit as comments
pointing at the canonical entry so manual BibTeX edits don't lose the
alternative names.
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEB_ROOT = HERE.parent
DATA = WEB_ROOT / "data" / "citations.json"
OUT = WEB_ROOT / "public" / "citations.bib"


def _bibtex_entry(key: str, e: dict[str, str]) -> str:
    title = e.get("title", "").replace("{", "").replace("}", "")
    fields: list[tuple[str, str]] = [("title", f"{{{title}}}")] if title else []
    # First-author guess from the key: ``kingma_ba_2015`` → ``Kingma and Ba``.
    parts = key.rsplit("_", 1)
    year = parts[-1] if parts[-1].isdigit() else None
    authors_part = parts[0] if year else key
    authors = " and ".join(a.capitalize() for a in authors_part.split("_") if a)
    if authors:
        fields.append(("author", f"{{{authors}}}"))
    if year:
        fields.append(("year", f"{{{year}}}"))
    if "arxiv" in e:
        fields.append(("eprint", f"{{{e['arxiv']}}}"))
        fields.append(("archivePrefix", "{arXiv}"))
        fields.append(("url", f"{{https://arxiv.org/abs/{e['arxiv']}}}"))
    if "doi" in e:
        fields.append(("doi", f"{{{e['doi']}}}"))
        fields.append(("url", f"{{https://doi.org/{e['doi']}}}"))
    if "url" in e and not any(f[0] == "url" for f in fields):
        fields.append(("url", f"{{{e['url']}}}"))

    body = ",\n".join(f"  {k:14s} = {v}" for k, v in fields)
    return f"@misc{{{key},\n{body}\n}}"


def main() -> int:
    if not DATA.is_file():
        print(f"[bibtex] {DATA} missing — nothing to do", file=sys.stderr)
        return 0
    raw = json.loads(DATA.read_text())
    # Drop the schema-doc field and any alias keys (we keep one entry
    # per unique arxiv/doi/url to avoid duplicate bib records).
    seen_signature: dict[tuple[str, ...], str] = {}
    canonical: dict[str, dict[str, str]] = {}
    aliases: dict[str, str] = {}  # alias key → canonical key

    for key, entry in raw.items():
        if key.startswith("_") or not isinstance(entry, dict):
            continue
        sig = (
            entry.get("arxiv", ""),
            entry.get("doi", ""),
            entry.get("url", ""),
            entry.get("title", ""),
        )
        if sig in seen_signature:
            aliases[key] = seen_signature[sig]
        else:
            seen_signature[sig] = key
            canonical[key] = entry

    lines: list[str] = [
        "% Lucid docs — exported BibTeX bibliography.",
        f"% Source: {DATA.relative_to(WEB_ROOT)}",
        f"% Entries: {len(canonical)} canonical + {len(aliases)} aliases.",
        "",
    ]
    for key in sorted(canonical):
        lines.append(_bibtex_entry(key, canonical[key]))
        # Append any aliases as comments so the next regeneration
        # preserves them visually for hand-editors.
        for alias_key, canon in aliases.items():
            if canon == key:
                lines.append(f"%   alias: {alias_key}")
        lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    rel = OUT.relative_to(WEB_ROOT)
    print(f"[bibtex] {len(canonical)} entries + {len(aliases)} aliases → {rel}")
    return 0


# Quiet ESLint-style unused-import warning for the imports we keep
# imported so the file reads as a complete script.
_ = re
_ = Path

if __name__ == "__main__":
    sys.exit(main())
