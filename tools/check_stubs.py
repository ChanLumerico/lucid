#!/usr/bin/env python3
"""Verify that generated .pyi stubs are up-to-date.

Regenerates stubs into a temporary location and diffs against the committed
versions. Fails if any stub is stale (i.e., needs re-running gen_pyi.py).

Exit 0 → stubs are current.
Exit 1 → stubs are stale; run  python tools/gen_pyi.py  to fix.
"""

from __future__ import annotations

import difflib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STUBS = {
    "engine.pyi": ROOT / "lucid" / "_C" / "engine.pyi",
    "tensor.pyi": ROOT / "lucid" / "_tensor" / "tensor.pyi",
    "__init__.pyi": ROOT / "lucid" / "__init__.pyi",
}


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_engine = f"{tmp}/engine.pyi"
        tmp_tensor = f"{tmp}/tensor.pyi"
        tmp_init   = f"{tmp}/__init__.pyi"

        result = subprocess.run(
            [
                sys.executable, "tools/gen_pyi.py",
                "--out-engine", tmp_engine,
                "--out-tensor", tmp_tensor,
                "--out-init",   tmp_init,
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if result.returncode != 0:
            print(f"[check_stubs] gen_pyi.py failed:\n{result.stderr}")
            return 1

        stale: list[str] = []
        for label, committed_path in STUBS.items():
            tmp_path = {"engine.pyi": tmp_engine, "tensor.pyi": tmp_tensor,
                        "__init__.pyi": tmp_init}[label]

            if not committed_path.exists():
                print(f"[check_stubs] MISSING  {committed_path.relative_to(ROOT)}")
                stale.append(label)
                continue

            committed = committed_path.read_text()
            generated = Path(tmp_path).read_text()

            if committed == generated:
                print(f"[check_stubs] OK       {committed_path.relative_to(ROOT)}")
            else:
                diff = list(difflib.unified_diff(
                    committed.splitlines(keepends=True),
                    generated.splitlines(keepends=True),
                    fromfile=f"{label} (committed)",
                    tofile=f"{label} (generated)",
                    n=2,
                ))
                print(f"[check_stubs] STALE    {committed_path.relative_to(ROOT)}")
                print("".join(diff[:40]))
                stale.append(label)

    if stale:
        print(f"\n[check_stubs] {len(stale)} stub(s) out of date. Run:")
        print("    python tools/gen_pyi.py")
        return 1

    print("[check_stubs] all stubs up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
