#!/usr/bin/env python3
"""tools/check_backend_layout.py — Enforce the split CpuBackend / GpuBackend layout (LCD-264).

Enforces three structural invariants on ``lucid/_C/backend/{cpu,gpu}/``:

1. ``inline-body`` / ``inline-body-too-long``:
   Inside ``CpuBackend.h`` and ``GpuBackend.h``, only members on the explicit
   ``KEEP_INLINE`` allow-list (tiny hot leaf helpers and class templates) may
   define an inline body in the class header, and every inline body must be at
   most ``MAX_INLINE_BODY_LINES`` lines (10 lines).

2. ``untracked-methods-cpp``:
   Because ``CMakeLists.txt`` collects ``backend/{cpu,gpu}/methods/*.cpp`` via
   ``file(GLOB ... CONFIGURE_DEPENDS)``, every ``.cpp`` file under ``methods/``
   must be tracked by git so an untracked scratch file is never linked into the
   engine.

3. ``methods-preamble`` / ``non-member-definition``:
   Every ``backend/{cpu,gpu}/methods/<Family>.cpp`` file must have a single
   ``#include "../<Backend>.h"`` directive before ``namespace lucid { namespace backend {``
   and contain only ``<Backend>::<member>(...)`` out-of-line member definitions
   (zero namespace-scope free functions or variables, eliminating ODR risk).
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

try:
    from tools import backend_split
except ImportError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import backend_split

ROOT = backend_split.ROOT
MAX_INLINE_BODY_LINES = 10


@dataclass(frozen=True)
class LayoutViolation:
    path: str
    line: int
    rule: str
    detail: str


def _git_tracked_methods(root: Path, rel_dir: str) -> set[str] | None:
    """Return git-tracked paths under ``rel_dir``, or ``None`` if ``root`` is not a git repo.

    Before the initial backend-split commit lands in ``HEAD`` (or on a synced
    remote checkout whose ``.git/index`` is not synced), if ``git ls-files``
    reports zero tracked files under ``rel_dir`` while every ``.cpp`` file in
    ``rel_dir`` is named ``<Family>.cpp`` for ``FAMILY_NAMES``, falls back to
    the canonical family set.
    """
    if not (root / ".git").exists():
        return None
    res = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--", rel_dir],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        return None
    tracked = {line.strip() for line in res.stdout.splitlines() if line.strip()}
    if not tracked:
        return {f"{rel_dir}/{fam}.cpp" for fam in backend_split.FAMILY_NAMES}
    return tracked


def scan_backend_layout(
    root: Path = ROOT,
    tracked_files: set[str] | None = None,
) -> list[LayoutViolation]:
    """Scan ``lucid/_C/backend/{cpu,gpu}`` and return any layout violations."""
    violations: list[LayoutViolation] = []

    for bk, (rel_header, class_name) in backend_split.BACKENDS.items():
        hdr_path = root / rel_header
        if not hdr_path.is_file():
            violations.append(
                LayoutViolation(rel_header, 1, "missing-header", "header file not found")
            )
            continue

        src = hdr_path.read_text(encoding="utf-8")
        try:
            parsed = backend_split.parse_header(src, class_name)
        except ValueError as exc:
            violations.append(LayoutViolation(rel_header, 1, "parse-error", str(exc)))
            continue

        keep_allowed = backend_split.KEEP_INLINE[bk]
        for m in parsed.members:
            if not m.is_function or not m.has_body:
                continue
            line = src.count("\n", 0, m.sig_start) + 1
            body_inner = backend_split.strip_noncode(
                src[m.body_open + 1 : m.seg_end - 1]
            )
            code_lines = sum(
                1 for ln in body_inner.splitlines() if ln.strip()
            )
            if m.name not in keep_allowed:
                violations.append(
                    LayoutViolation(
                        rel_header,
                        line,
                        "inline-body",
                        f"{class_name}::{m.name} has an inline body in {rel_header}; "
                        f"move its definition to lucid/_C/backend/{bk}/methods/<Family>.cpp",
                    )
                )
            elif code_lines > MAX_INLINE_BODY_LINES:
                violations.append(
                    LayoutViolation(
                        rel_header,
                        line,
                        "inline-body-too-long",
                        f"{class_name}::{m.name} inline body is {code_lines} lines "
                        f"(cap: {MAX_INLINE_BODY_LINES})",
                    )
                )

        rel_methods_dir = f"lucid/_C/backend/{bk}/methods"
        methods_dir = root / rel_methods_dir
        if not methods_dir.is_dir():
            violations.append(
                LayoutViolation(
                    rel_methods_dir,
                    1,
                    "missing-methods-dir",
                    f"expected directory {rel_methods_dir}",
                )
            )
            continue

        git_set = (
            tracked_files
            if tracked_files is not None
            else _git_tracked_methods(root, rel_methods_dir)
        )
        expected_hdr = hdr_path.name
        for cpp_file in sorted(methods_dir.glob("*.cpp")):
            rel_cpp = cpp_file.relative_to(root).as_posix()
            if git_set is not None and rel_cpp not in git_set:
                violations.append(
                    LayoutViolation(
                        rel_cpp,
                        1,
                        "untracked-methods-cpp",
                        f"{rel_cpp} is not tracked by git (GLOB would link stray TU)",
                    )
                )
            cpp_src = cpp_file.read_text(encoding="utf-8")
            try:
                backend_split.parse_methods_cpp(
                    cpp_src, rel_cpp, class_name, expected_hdr
                )
            except ValueError as exc:
                msg = str(exc)
                rule = (
                    "non-member-definition"
                    if f"non-{class_name} definition" in msg
                    or "unexpected top-level ';'" in msg
                    else "methods-preamble"
                )
                violations.append(LayoutViolation(rel_cpp, 1, rule, msg))

    return violations


def check(root: Path = ROOT, quiet: bool = False) -> int:
    violations = scan_backend_layout(root=root)
    if not violations:
        if not quiet:
            print("[check_backend_layout] OK — CpuBackend/GpuBackend layout clean.")
        return 0
    for v in violations:
        print(
            f"[check_backend_layout] {v.path}:{v.line}: [{v.rule}] {v.detail}",
            file=sys.stderr,
        )
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    return check(root=args.root.resolve(), quiet=args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
