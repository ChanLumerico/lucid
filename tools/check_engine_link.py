"""Check that the built engine links the MLX build it ships, before anything loads it.

The engine imports ~190 MLX C++ symbols whose mangled names change between
MLX patch releases (LCD-292).  A stale build tree, a pin bumped without a
rebuild, or a wheel assembled from mismatched parts loads fine on the machine
that built it and dies on the user's with dyld's "Symbol not found".  This
reads the binaries instead of loading them:

* the engine links ``@rpath/liblucid_mlx.dylib`` and never ``libmlx.dylib``
  (the name a Python ``mlx`` loads, which dyld would hand to both);
* every MLX symbol the engine imports is exported by the ``liblucid_mlx``
  its rpath resolves to, and none is left to flat lookup, where it could bind
  to a Python ``mlx``'s library instead;
* ``liblucid_mlx`` is built for the pinned macOS floor (``[tool.lucid.mlx]
  macos``), the floor ``lucid._C`` enforces at import, and the engine for
  that floor or a newer macOS (a developer's own deployment target);
* with ``--wheel``, the engine targets exactly the floor, its only rpath is
  ``@loader_path`` and the MLX runtime files sit beside it.  A build-machine path in a wheel passes every
  smoke test on the build machine and nowhere else, as in the 3.0.0 release.

Usage::

    python3 tools/check_engine_link.py                  # the checkout's engine
    python3 tools/check_engine_link.py ENGINE --wheel   # an installed wheel's

Exits 1 with one line per problem, 0 when the link is sound.
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import fetch_mlx

LOADER_PATH = "@loader_path"
_UNDEFINED = re.compile(r"\(undefined\) (?:weak )?external (\S+) \((?:from )?([^)]*)\)")


# Seconds an otool / nm / c++filt run may take; the engine's nm is ~1 s.
TIMEOUT = 120.0


@dataclass(frozen=True)
class LoadCommands:
    rpaths: list[str]
    minos: str


def _tool(*command: str, stdin: str | None = None) -> str:
    return fetch_mlx.run(*command, stdin=stdin, timeout=TIMEOUT)


def load_commands(binary: Path) -> LoadCommands:
    text = _tool("otool", "-l", str(binary))
    rpaths = re.findall(r"cmd LC_RPATH\n\s+cmdsize \d+\n\s+path (.+?) \(offset", text)
    return LoadCommands(rpaths, fetch_mlx.minos(binary, timeout=TIMEOUT))


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def linked_libraries(binary: Path) -> list[str]:
    """The install names *binary* loads (``otool -L``), its own id excluded."""
    lines = _tool("otool", "-L", str(binary)).splitlines()[1:]
    names = [line.strip().split(" (compatibility")[0] for line in lines]
    own = _tool("otool", "-D", str(binary)).splitlines()[1:]
    return [n for n in names if n and n not in own]


def undefined_symbols(binary: Path) -> dict[str, str]:
    """Each symbol *binary* imports -> the library it is bound to (``nm -m``)."""
    found = {}
    for line in _tool("nm", "-m", "-u", str(binary)).splitlines():
        match = _UNDEFINED.search(line)
        if match:
            found[match.group(1)] = match.group(2)
    return found


def exported_symbols(dylib: Path) -> set[str]:
    return set(_tool("nm", "-gUj", str(dylib)).split())


def demangle(symbols: list[str]) -> list[str]:
    out = _tool("c++filt", stdin="\n".join(symbols) + "\n").splitlines()
    if len(out) != len(symbols):
        raise fetch_mlx.FetchError(f"c++filt returned {len(out)} names for {len(symbols)} symbols")
    return out


def resolve_mlx(engine: Path, rpaths: list[str]) -> Path | None:
    """The liblucid_mlx dyld would load for *engine*: the first rpath that has it."""
    for rpath in rpaths:
        directory = Path(rpath.replace(LOADER_PATH, str(engine.parent), 1))
        candidate = directory / fetch_mlx.MLX_LIB
        if candidate.is_file():
            return candidate
    return None


def _check_symbols(engine: Path, mlx: Path) -> list[str]:
    imports = undefined_symbols(engine)
    names = sorted(imports)
    mlx_bound = Path(fetch_mlx.MLX_LIB).stem
    wanted = {
        name
        for name, readable in zip(names, demangle(names), strict=True)
        if "mlx::" in readable or imports[name] == mlx_bound
    }
    if not wanted:
        return [f"{engine.name}: imports no MLX symbol — not an MLX-linked engine?"]
    problems = []
    stray = sorted(n for n in wanted if imports[n] != mlx_bound)
    if stray:
        problems.append(
            f"{engine.name}: {len(stray)} MLX symbol(s) not bound to {mlx_bound}, "
            f"e.g. {stray[0]} ({imports[stray[0]]})"
        )
    missing = sorted(wanted - exported_symbols(mlx))
    if missing:
        sample = "\n    ".join(demangle(missing[:5]))
        problems.append(
            f"{mlx}: lacks {len(missing)} of the {len(wanted)} MLX symbols "
            f"{engine.name} imports (stale build or mismatched pin), e.g.\n    {sample}"
        )
    return problems


def _check_wheel_layout(engine: Path, commands: LoadCommands) -> list[str]:
    problems = []
    if commands.rpaths != [LOADER_PATH]:
        problems.append(
            f"{engine.name}: rpaths {commands.rpaths}, a wheel's must be [{LOADER_PATH!r}] "
            "(an absolute rpath names the build machine)"
        )
    for name in (fetch_mlx.MLX_LIB, fetch_mlx.JACCL_LIB, fetch_mlx.METALLIB):
        if not (engine.parent / name).is_file():
            problems.append(f"{engine.parent}: no {name} beside the engine")
    return problems


def check(engine: Path, floor: str, wheel: bool = False) -> list[str]:
    """Every problem with *engine*'s link to its MLX, as one line each."""
    problems = []
    libraries = linked_libraries(engine)
    wanted = f"@rpath/{fetch_mlx.MLX_LIB}"
    if wanted not in libraries:
        problems.append(f"{engine.name}: does not link {wanted}")
    problems += [
        f"{engine.name}: links {lib}, the library a Python mlx loads"
        for lib in libraries
        if Path(lib).name in ("libmlx.dylib", "libjaccl.dylib")
    ]
    commands = load_commands(engine)
    # A release must target exactly the floor lucid._C enforces; a developer's
    # own build may target a newer macOS than that.
    if wheel and commands.minos != floor:
        problems.append(f"{engine.name}: minos {commands.minos}, a wheel's must be {floor}")
    elif _version(commands.minos) < _version(floor):
        problems.append(f"{engine.name}: minos {commands.minos}, below the pinned floor {floor}")
    if wheel:
        problems += _check_wheel_layout(engine, commands)
    mlx = resolve_mlx(engine, commands.rpaths)
    if mlx is None:
        problems.append(
            f"{engine.name}: no {fetch_mlx.MLX_LIB} on its rpaths {commands.rpaths} "
            "(python3 tools/fetch_mlx.py stages it)"
        )
        return problems
    # The floor is defined as this binary's minos (fetch_mlx --bump records it).
    mlx_minos = fetch_mlx.minos(mlx, timeout=TIMEOUT)
    if mlx_minos != floor:
        problems.append(f"{mlx}: minos {mlx_minos}, pinned floor {floor}")
    return problems + _check_symbols(engine, mlx)


def default_engine() -> Path | None:
    return next(iter(sorted((ROOT / "lucid/_C").glob("engine*.so"))), None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("engine", nargs="?", type=Path, help="engine*.so (default: the checkout's)")
    parser.add_argument(
        "--wheel", action="store_true", help="also require the self-contained wheel layout"
    )
    args = parser.parse_args(argv)
    engine = args.engine or default_engine()
    if engine is None or not engine.is_file():
        print(f"check_engine_link: no engine at {engine or ROOT / 'lucid/_C'}", file=sys.stderr)
        return 1
    try:
        problems = check(engine.resolve(), fetch_mlx.read_pin().macos, wheel=args.wheel)
    except fetch_mlx.FetchError as exc:
        problems = [str(exc)]
    for problem in problems:
        print(f"check_engine_link: {problem}", file=sys.stderr)
    if not problems:
        print(f"check_engine_link: {engine} links its MLX soundly")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
