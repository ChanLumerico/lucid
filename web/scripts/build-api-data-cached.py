"""Cached wrapper around build-api-data.py.

Skips the (slow) Griffe walk when no Python source under lucid/ — nor the
build script itself — has changed since the last successful build.  Cache
key is the set of (relative-path, mtime_ns) pairs for every tracked file,
stored as a single line in public/api-data/.cache-key.

Env vars:
  SKIP_API_BUILD=1    force skip even on cache miss
  FORCE_API_BUILD=1   force rebuild even on cache hit

Exit code mirrors the underlying build-api-data.py.
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEB_ROOT = HERE.parent
REPO_ROOT = WEB_ROOT.parent
LUCID_SRC = REPO_ROOT / "lucid"
OUT_DIR = WEB_ROOT / "public" / "api-data"
BUILD_SCRIPT = HERE / "build-api-data.py"
CACHE_FILE = OUT_DIR / ".cache-key"


def _compute_key() -> str:
    """Hash (path, mtime_ns) for every .py under lucid/ plus the build script."""
    h = hashlib.sha256()
    sources: list[Path] = sorted(LUCID_SRC.rglob("*.py"))
    sources.append(BUILD_SCRIPT)
    for p in sources:
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        rel = p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p
        h.update(f"{rel}:{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def _outputs_exist() -> bool:
    return OUT_DIR.is_dir() and any(OUT_DIR.glob("*.json"))


def main() -> int:
    if os.environ.get("SKIP_API_BUILD") == "1":
        print("[api-data] SKIP_API_BUILD=1 — skipping build")
        return 0

    force = os.environ.get("FORCE_API_BUILD") == "1"
    current_key = _compute_key()

    if not force and _outputs_exist() and CACHE_FILE.is_file():
        try:
            cached = CACHE_FILE.read_text().strip()
        except OSError:
            cached = ""
        if cached == current_key:
            print("[api-data] cache hit — skipping rebuild")
            return 0

    if force:
        print("[api-data] FORCE_API_BUILD=1 — rebuilding")
    else:
        print("[api-data] cache miss — rebuilding")

    result = subprocess.run([sys.executable, str(BUILD_SCRIPT)])
    if result.returncode == 0:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(_compute_key())
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
