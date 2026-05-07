#!/usr/bin/env python3
"""
vault_sync_check.py — fired as a PostToolUse hook after Edit/Write/MultiEdit.

Reads JSON from stdin (Claude Code hook payload), maps the edited file path to
the obsidian vault note(s) that should be kept in sync, and emits a structured
reminder via stderr (which Claude Code surfaces back to the agent as a
system-reminder, so it cannot be silently ignored).

The script is intentionally read-only: it never edits files. It only nudges the
agent to update vault notes in the same task.

Maintenance: when adding new public-API areas, append to ``MAPPING`` below.
"""

from __future__ import annotations  # local script — not part of lucid package

import fnmatch
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# fmt: off
# Map: glob pattern (relative to repo root) → list of vault notes to keep in sync.
# Order matters only for clarity; all matching patterns contribute.
MAPPING: list[tuple[str, list[str]]] = [
    # ── C++ engine surface ────────────────────────────────────────────────
    ("lucid/_C/ops/**",        ["obsidian/api/api-cpp-tree.md"]),
    ("lucid/_C/bindings/**",   ["obsidian/api/api-cpp-tree.md"]),
    ("lucid/_C/engine.pyi",    ["obsidian/api/api-cpp-tree.md"]),
    ("lucid/_C/registry/**",   ["obsidian/api/api-cpp-tree.md"]),
    ("lucid/_C/core/**",       ["obsidian/api/api-cpp-tree.md"]),
    ("lucid/_C/tensor/**",     ["obsidian/api/api-cpp-tree.md"]),

    # ── Top-level Python ──────────────────────────────────────────────────
    ("lucid/__init__.py",      ["obsidian/api/api-python-toplevel.md"]),
    ("lucid/__init__.pyi",     ["obsidian/api/api-python-toplevel.md"]),
    ("lucid/_ops/**",          ["obsidian/api/api-python-toplevel.md"]),
    ("lucid/_factories/**",    ["obsidian/api/api-python-toplevel.md"]),
    ("lucid/_dtype.py",        ["obsidian/api/api-python-toplevel.md"]),
    ("lucid/_device.py",       ["obsidian/api/api-python-toplevel.md"]),

    # ── Tensor class ──────────────────────────────────────────────────────
    ("lucid/_tensor/**",       ["obsidian/api/api-python-tensor.md"]),

    # ── nn ────────────────────────────────────────────────────────────────
    ("lucid/nn/__init__.py",   ["obsidian/api/api-python-nn.md"]),
    ("lucid/nn/module.py",     ["obsidian/api/api-python-nn.md"]),
    ("lucid/nn/parameter.py",  ["obsidian/api/api-python-nn.md"]),
    ("lucid/nn/hooks.py",      ["obsidian/api/api-python-nn.md"]),
    ("lucid/nn/_state_dict.py",["obsidian/api/api-python-nn.md"]),
    ("lucid/nn/modules/**",    ["obsidian/api/api-python-nn.md"]),
    ("lucid/nn/functional/**", ["obsidian/api/api-python-nn-functional.md"]),
    ("lucid/nn/init.py",       ["obsidian/api/api-python-nn-init.md"]),
    ("lucid/nn/utils/**",      ["obsidian/api/api-python-nn-utils.md"]),

    # ── optim / autograd / linalg ─────────────────────────────────────────
    ("lucid/optim/**",         ["obsidian/api/api-python-optim.md"]),
    ("lucid/autograd/**",      ["obsidian/api/api-python-autograd.md"]),
    ("lucid/linalg/**",        ["obsidian/api/api-python-linalg.md"]),

    # ── utils / amp / profiler / einops / serialization ───────────────────
    ("lucid/utils/data/**",    ["obsidian/api/api-python-utils-data.md"]),
    ("lucid/amp/**",           ["obsidian/api/api-python-misc.md"]),
    ("lucid/profiler/**",      ["obsidian/api/api-python-misc.md"]),
    ("lucid/einops/**",        ["obsidian/api/api-python-misc.md"]),
    ("lucid/serialization/**", ["obsidian/api/api-python-misc.md"]),
]

# Phase / milestone signals (touch these → consider retro/ note)
RETRO_SIGNALS = ["lucid/version.py", "CHANGELOG.md"]


def _resolve_rel(file_path: str) -> str | None:
    """Return repo-relative path string if inside the repo, else None."""
    if not file_path:
        return None
    try:
        abs_path = Path(file_path).resolve()
        rel = abs_path.relative_to(REPO_ROOT)
    except (ValueError, OSError):
        return None
    return str(rel)


def _match(rel: str) -> tuple[set[str], bool]:
    """Return (vault notes to update, retro signal hit)."""
    notes: set[str] = set()
    for pattern, target_notes in MAPPING:
        if fnmatch.fnmatch(rel, pattern):
            notes.update(target_notes)
    retro = any(fnmatch.fnmatch(rel, p) for p in RETRO_SIGNALS)
    return notes, retro


def main() -> int:
    raw = sys.stdin.read().strip()
    if not raw:
        return 0
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return 0

    tool_input = payload.get("tool_input") or {}
    file_path = (
        tool_input.get("file_path")
        or tool_input.get("path")
        or ""
    )
    rel = _resolve_rel(file_path)
    if rel is None:
        return 0

    # Editing a vault file itself: never remind (avoid recursion / noise).
    if rel.startswith("obsidian/") or rel == "CLAUDE.md":
        return 0

    notes, retro = _match(rel)
    if not notes and not retro:
        return 0

    lines = [
        "📝 Vault sync reminder",
        f"   File touched: `{rel}`",
        "",
    ]
    if notes:
        lines.append(
            "If this edit changed any public API surface (new/removed/renamed "
            "export, signature change, behavior change), update the following "
            "vault note(s) **in this same task**:"
        )
        for note in sorted(notes):
            lines.append(f"  • {note}")
        lines += [
            "",
            "Also bump the `Last verified` line at the top of each touched "
            "note to today's date + current commit SHA (run `git rev-parse "
            "--short HEAD`).",
        ]
    if retro:
        lines += [
            "",
            "🏁 Phase / milestone signal detected — consider writing a "
            "`obsidian/retro/retro-<topic>.md` note if a meaningful chunk "
            "of work just landed.",
        ]
    lines.append("")
    lines.append(
        "If this edit was internal-only (no public API change), this "
        "reminder may be safely ignored — but explicitly say so in your "
        "next message so the user knows you considered it."
    )

    # PostToolUse hook output: emit JSON on stdout so Claude Code injects
    # the message as additionalContext into the model. This is the
    # documented control channel; stderr is unreliable across versions.
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": "\n".join(lines),
        }
    }
    sys.stdout.write(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
