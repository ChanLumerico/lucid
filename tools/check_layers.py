#!/usr/bin/env python3
"""Validate Lucid C++ layer include dependencies.

The rule is intentionally simple: a higher layer may include lower layers, but
lower layers may not include higher layers. Third-party/system includes are
ignored. Directories that are planned but not present yet are included in the
map so the check tightens automatically as the refactor progresses.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')


@dataclass(frozen=True)
class Layer:
    name: str
    rank: int


LAYER_BY_TOP = {
    "core": Layer("core", 0),
    "tensor": Layer("tensor", 1),
    "backend": Layer("backend", 2),
    "autograd": Layer("autograd", 3),
    "kernel": Layer("kernel", 4),
    "registry": Layer("registry", 0),
    "ops": Layer("ops", 6),
    "nn": Layer("ops", 6),  # Pre Phase 4 move; treated as op implementations.
    "optim": Layer("optim", 7),
    "random": Layer("random", 7),
    "bindings": Layer("bindings", 8),
}

TEMPORARY_ALLOWLIST = {
    (
        Path("lucid/_C/core/TensorImpl.cpp"),
        Path("lucid/_C/backend/gpu/MlxBridge.h"),
    ): "TensorImpl still owns GPU upload/download until tensor/ extraction in Phase 2.",
}


def layer_for(path: Path) -> Layer | None:
    try:
        rel = path.relative_to(Path("lucid/_C"))
    except ValueError:
        return None
    if not rel.parts:
        return None
    return LAYER_BY_TOP.get(rel.parts[0])


def resolve_include(source: Path, include: str) -> Path | None:
    candidates = [
        (source.parent / include).resolve(),
        (Path("lucid/_C") / include).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def is_allowed(source_layer: Layer, target_layer: Layer) -> bool:
    if source_layer.name == target_layer.name:
        return True

    # Registry is intentionally orthogonal: schema metadata may depend only on
    # core, and high layers may consume it. The current code still keeps
    # OpRegistry/OpSchema in core; this branch makes a future registry/
    # directory obey the target contract as soon as it appears.
    if source_layer.name == "registry":
        return target_layer.name == "core"
    if target_layer.name == "registry":
        return source_layer.rank >= 4

    return source_layer.rank >= target_layer.rank


def is_temporarily_allowed(source: Path, target: Path) -> bool:
    return (source, target) in TEMPORARY_ALLOWLIST


def iter_sources(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.suffix in {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"}
    )


def check(root: Path) -> list[str]:
    errors: list[str] = []
    cwd = Path.cwd().resolve()
    for source_abs in iter_sources(root):
        source = source_abs.relative_to(cwd)
        source_layer = layer_for(source)
        if source_layer is None:
            continue
        try:
            lines = source_abs.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = source_abs.read_text(encoding="latin-1").splitlines()

        for lineno, line in enumerate(lines, start=1):
            match = INCLUDE_RE.match(line)
            if not match:
                continue
            target_abs = resolve_include(source, match.group(1))
            if target_abs is None:
                continue
            try:
                target = target_abs.relative_to(cwd)
            except ValueError:
                continue
            target_layer = layer_for(target)
            if target_layer is None:
                continue
            if is_temporarily_allowed(source, target):
                continue
            if not is_allowed(source_layer, target_layer):
                errors.append(
                    f"{source}:{lineno}: {source_layer.name} must not include "
                    f"{target_layer.name}: {match.group(1)}"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="lucid/_C", help="C++ source root")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"error: source root does not exist: {root}", file=sys.stderr)
        return 2

    os.chdir(root.parents[1])
    errors = check(Path(args.root).resolve())
    if errors:
        print("Layer dependency violations:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    print("Layer dependency check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
