#!/usr/bin/env python3
"""
tools/check_stubs.py — Verify that committed .pyi stubs are up to date.

Runs gen_pyi.py to a temp directory and diffs output against committed stubs.
Exits non-zero if any stub is stale.
"""

import sys
import os
import difflib
import tempfile
import importlib
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _run_gen_pyi_to_tempdir() -> dict[str, str]:
    """Run gen_pyi logic and return {relative_path: content}."""
    import tools.gen_pyi as gen

    engine_content, _ = gen.gen_engine_pyi()
    tensor_content, _ = gen.gen_tensor_pyi()
    init_content, _ = gen.gen_init_pyi()

    return {
        "lucid/_C/engine.pyi": engine_content,
        "lucid/_tensor/tensor.pyi": tensor_content,
        "lucid/__init__.pyi": init_content,
    }


def main() -> int:
    stale: list[str] = []

    try:
        generated = _run_gen_pyi_to_tempdir()
    except Exception as exc:
        print(f"[check_stubs] ERROR running gen_pyi: {exc}", file=sys.stderr)
        return 1

    for rel_path, new_content in generated.items():
        committed_path = ROOT / rel_path
        if not committed_path.exists():
            print(f"[check_stubs] MISSING   {rel_path}")
            stale.append(rel_path)
            continue

        committed = committed_path.read_text(encoding="utf-8")
        if committed == new_content:
            print(f"[check_stubs] OK       {rel_path}")
        else:
            diff = list(
                difflib.unified_diff(
                    committed.splitlines(keepends=True),
                    new_content.splitlines(keepends=True),
                    fromfile=f"{rel_path} (committed)",
                    tofile=f"{rel_path} (generated)",
                    n=3,
                )
            )
            print(f"[check_stubs] STALE    {rel_path}")
            sys.stdout.writelines(diff[:40])
            if len(diff) > 40:
                print(f"  ... ({len(diff) - 40} more diff lines)")
            stale.append(rel_path)

    if stale:
        print(
            f"\n[check_stubs] {len(stale)} stub(s) out of date. Run:\n"
            "    python tools/gen_pyi.py\n"
            f"    git add {' '.join(stale)}\n",
            file=sys.stderr,
        )
        print(
            "\n  Commit blocked. Run the following to fix, then re-stage and commit:\n"
        )
        print("    python tools/gen_pyi.py")
        print(
            f"    git add lucid/_C/engine.pyi lucid/_tensor/tensor.pyi lucid/__init__.pyi"
        )
        return 1

    print("[check_stubs] all stubs up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
