"""Check a built Lucid wheel's contents, tag and size before it is published.

The wheel carries its own MLX (LCD-292): ``liblucid_mlx.dylib``,
``liblucid_jaccl.dylib`` and ``mlx.metallib`` beside ``lucid/_C/engine*.so``,
found through the engine's ``@loader_path`` rpath.  A wheel without them
imports nowhere; a wheel with ``libmlx.dylib`` in it carries the name a
Python ``mlx`` loads, which dyld would share between the two.

The metallib is most of the wheel, and it grows with every MLX release
(+48% from 0.31.0 to 0.32.3).  PyPI refuses a file over 100 MiB unless the
project's limit is raised, so the gate fails there and warns from 90 MiB —
while there is still time to ask for more.

Usage: ``python3 tools/check_wheel.py dist/lucid_dl-*.whl``
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import fetch_mlx

MIB = 1 << 20
PYPI_LIMIT = 100 * MIB
WARN_AT = 90 * MIB
ENGINE_DIR = "lucid/_C/"
SHIPPED = (fetch_mlx.MLX_LIB, fetch_mlx.JACCL_LIB, fetch_mlx.METALLIB)
FOREIGN = ("libmlx.dylib", "libjaccl.dylib")


def platform_tag(floor: str) -> str:
    """The tag a wheel for *floor* carries: pip spells every macOS 11+ as ``X_0``."""
    return f"macosx_{floor.split('.')[0]}_0_arm64"


def check(wheel: Path, floor: str) -> tuple[list[str], list[str]]:
    """``(problems, warnings)`` for *wheel*; an empty first list means publishable."""
    problems: list[str] = []
    warnings: list[str] = []
    tag = wheel.name.removesuffix(".whl").split("-")[-1]
    if tag != platform_tag(floor):
        problems.append(f"platform tag {tag}, expected {platform_tag(floor)}")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    engines = [n for n in names if n.startswith(f"{ENGINE_DIR}engine") and n.endswith(".so")]
    if len(engines) != 1:
        problems.append(f"{len(engines)} engine binaries under {ENGINE_DIR}, expected 1")
    problems += [
        f"no {ENGINE_DIR}{name} (the engine's bundled MLX)"
        for name in SHIPPED
        if f"{ENGINE_DIR}{name}" not in names
    ]
    problems += [
        f"ships {name}: a Python mlx's library name, which dyld would share"
        for name in names
        if Path(name).name in FOREIGN
    ]
    size = wheel.stat().st_size
    if size > PYPI_LIMIT:
        problems.append(f"{size / MIB:.1f} MiB, over PyPI's {PYPI_LIMIT // MIB} MiB file limit")
    elif size > WARN_AT:
        warnings.append(
            f"{size / MIB:.1f} MiB, within {(PYPI_LIMIT - size) / MIB:.1f} MiB of PyPI's "
            f"{PYPI_LIMIT // MIB} MiB file limit: request a larger limit for lucid-dl"
        )
    return problems, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args(argv)
    if not args.wheel.is_file():
        parser.error(f"{args.wheel}: no such wheel")
    try:
        problems, warnings = check(args.wheel, fetch_mlx.read_pin().macos)
    except (zipfile.BadZipFile, fetch_mlx.FetchError) as exc:
        problems, warnings = [str(exc)], []
    # GitHub renders ::warning:: lines as annotations on the run.
    prefix = "::warning::" if os.environ.get("GITHUB_ACTIONS") == "true" else "warning: "
    for warning in warnings:
        print(f"{prefix}check_wheel: {args.wheel.name}: {warning}")
    for problem in problems:
        print(f"check_wheel: {args.wheel.name}: {problem}", file=sys.stderr)
    if not problems:
        size = args.wheel.stat().st_size / MIB
        print(f"check_wheel: {args.wheel.name} ok ({size:.1f} MiB, bundles {', '.join(SHIPPED)})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
