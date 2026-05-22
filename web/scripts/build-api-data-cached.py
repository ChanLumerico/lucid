"""Cached wrapper around build-api-data.py.

Two-tier caching:

  * **Global key** — whole-tree fingerprint over every ``.py`` under
    ``lucid/`` plus the build script itself.  Hit → skip everything,
    no Griffe walk, no JSON writes.  Used for the common "I'm editing
    docs site code" loop where the Python tree is untouched.

  * **Per-module keys** — for each emitted module slug, hash the
    Python files the slug's JSON depends on.  When the global key
    misses, we still skip rebuilding individual JSONs whose
    fingerprints match.  Cuts a typical "I edited one lucid file" loop
    from ~9 min to ~30 s.

Cache files live under ``public/api-data/.cache/``:
  * ``key`` — global fingerprint (string).
  * ``modules.json`` — ``{slug: {hash, sources: [files], deps: [deps]}}``.

Env vars:
  SKIP_API_BUILD=1    force skip even on cache miss
  FORCE_API_BUILD=1   force full rebuild, ignore both caches
  FORCE_FULL_REBUILD=1  alias of FORCE_API_BUILD

Exit code mirrors the underlying build-api-data.py.
"""

import hashlib
import json
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
CACHE_DIR = OUT_DIR / ".cache"
CACHE_FILE = OUT_DIR / ".cache-key"
MODULE_CACHE = CACHE_DIR / "modules.json"


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


def _slug_to_source_dir(slug: str) -> Path | None:
    """Map a slug (``lucid.nn.functional``) to its directory under lucid/.

    Used by the per-module cache to fingerprint only the files that
    contribute to that slug's JSON — touching ``lucid/optim/sgd.py``
    shouldn't invalidate the ``lucid.nn.functional`` slug's cache.
    Returns ``None`` for synthesized slugs (``lucid.ops`` /
    ``lucid.creation`` / ``lucid.ops.composite``) — those depend on
    the whole lucid tree and fall back to the global key.
    """
    if slug in ("lucid.ops", "lucid.creation", "lucid.ops.composite"):
        return None
    # ``lucid.tensor`` reads from lucid/_tensor/tensor.py.
    if slug == "lucid.tensor":
        return LUCID_SRC / "_tensor"
    # Synthetic ``lucid._C.engine`` depends on engine.pyi + bindings —
    # safer to treat as whole-tree.
    if slug == "lucid._C.engine":
        return None
    # Strip leading ``lucid.``, swap dots for slashes.
    rel = slug.removeprefix("lucid.").replace(".", "/")
    candidate = LUCID_SRC / rel
    if candidate.is_dir():
        return candidate
    py = candidate.with_suffix(".py")
    if py.is_file():
        return py
    return None


def _hash_paths(paths: list[Path]) -> str:
    """sha256 over (rel path, mtime_ns) pairs — same shape as the global key."""
    h = hashlib.sha256()
    for p in sorted(paths):
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        rel = p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p
        h.update(f"{rel}:{st.st_mtime_ns}\n".encode())
    h.update(f"build_script:{BUILD_SCRIPT.stat().st_mtime_ns}\n".encode())
    return h.hexdigest()


def _per_module_fingerprints() -> dict[str, str]:
    """Compute a fingerprint per slug, keyed by slug.

    Walks each slug's source directory recursively.  Synth slugs return
    no fingerprint (they always rebuild when the global key misses).
    Used for the post-build cache write — the actual rebuild decision
    happens module-by-module in :func:`_should_skip_slug`.
    """
    # We need the list of emitted slugs.  Easiest source: the existing
    # *.json files under OUT_DIR — they were just emitted, so they're
    # the canonical list.
    result: dict[str, str] = {}
    if not OUT_DIR.is_dir():
        return result
    for jp in OUT_DIR.glob("*.json"):
        if jp.name.startswith("_") or jp.name.startswith("."):
            continue
        slug = jp.stem
        d = _slug_to_source_dir(slug)
        if d is None:
            continue
        files = list(d.rglob("*.py")) if d.is_dir() else [d]
        result[slug] = _hash_paths(files)
    return result


def _load_module_cache() -> dict[str, str]:
    if not MODULE_CACHE.is_file():
        return {}
    try:
        return json.loads(MODULE_CACHE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_module_cache(fingerprints: dict[str, str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODULE_CACHE.write_text(json.dumps(fingerprints, indent=2, sort_keys=True))


def main() -> int:
    if os.environ.get("SKIP_API_BUILD") == "1":
        print("[api-data] SKIP_API_BUILD=1 — skipping build")
        return 0

    force = (
        os.environ.get("FORCE_API_BUILD") == "1"
        or os.environ.get("FORCE_FULL_REBUILD") == "1"
    )
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
        print("[api-data] FORCE rebuild")
    else:
        print("[api-data] global cache miss — checking per-module cache…")

    # ── Per-module cache — only invoke build-api-data.py for slugs
    # whose source-dir fingerprint has changed.  The build script
    # already supports ``--module <slug>`` so we drive it once per
    # stale slug.  Synth slugs and lucid._C.engine fall back to a
    # whole-tree rebuild when *anything* changes.
    prev_fingerprints = _load_module_cache() if not force else {}
    current_per_module: dict[str, str] = {}
    if OUT_DIR.is_dir():
        for jp in OUT_DIR.glob("*.json"):
            if jp.name.startswith("_") or jp.name.startswith("."):
                continue
            slug = jp.stem
            d = _slug_to_source_dir(slug)
            if d is None:
                continue
            files = list(d.rglob("*.py")) if d.is_dir() else [d]
            current_per_module[slug] = _hash_paths(files)

    # Decide rebuild strategy.  If we have NO prior cache or the per-
    # module cache is empty, fall back to a full rebuild — it's faster
    # than scheduling 76 ``--module`` invocations on a cold start.
    stale_slugs: list[str] = []
    needs_full = force or not prev_fingerprints
    if not needs_full:
        for slug, fp in current_per_module.items():
            if prev_fingerprints.get(slug) != fp:
                stale_slugs.append(slug)
        # Synth slugs depend on global tree — rebuild them whenever
        # ANY manifest slug is stale (cheap, the source data is
        # already loaded into memory at that point anyway).
        if stale_slugs:
            for synth in ("lucid.ops", "lucid.creation", "lucid.ops.composite"):
                if synth not in stale_slugs and (OUT_DIR / f"{synth}.json").exists():
                    stale_slugs.append(synth)

    if needs_full:
        print("[api-data] running full rebuild")
        result = subprocess.run([sys.executable, str(BUILD_SCRIPT)])
    elif not stale_slugs:
        # Per-module cache is fully warm — global mtime changed (maybe
        # the build script itself) but no module source changed.  Just
        # refresh the global key and exit.
        print("[api-data] per-module cache fully warm — skipping rebuild")
        CACHE_FILE.write_text(current_key)
        return 0
    else:
        print(f"[api-data] rebuilding {len(stale_slugs)} stale slug(s): "
              f"{', '.join(stale_slugs[:5])}"
              + (f" + {len(stale_slugs) - 5} more" if len(stale_slugs) > 5 else ""))
        result = subprocess.run([
            sys.executable, str(BUILD_SCRIPT),
            "--slugs", *stale_slugs,
        ])

    if result.returncode == 0:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(_compute_key())
        # Refresh per-module fingerprints from what's actually on disk
        # — captures any slugs the build emitted, regardless of which
        # branch we took above.
        _save_module_cache(_per_module_fingerprints())
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
