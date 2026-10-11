"""Fetch, verify and stage the MLX build that Lucid's engine links.

The engine does not link whatever ``mlx`` happens to be installed.  It links
one pinned build of Apple MLX, recorded in ``pyproject.toml`` under
``[tool.lucid.mlx]`` (version, wheel file name, sha256, URL, macOS floor),
and ships it inside the Lucid wheel next to ``engine*.so``.  MLX's C++ ABI
changes between patch releases, so this is the only way the engine keeps
working beside any ``mlx`` version a user installs.

Staging takes the official ``mlx-metal`` wheel and produces::

    <cache>/mlx-<version>-<sha12>/
        include/                    MLX, metal-cpp and jaccl headers
        lib/liblucid_mlx.dylib      libmlx.dylib, install name @rpath/liblucid_mlx.dylib
        lib/liblucid_jaccl.dylib    libjaccl.dylib, loaded by liblucid_mlx via @loader_path
        lib/mlx.metallib            unchanged; MLX finds it next to its dylib (dladdr)
        LICENSE                     MLX's license, from the wheel's dist-info
        .lucid-staged               written last; its presence marks a complete entry

The libraries are renamed only so dyld does not mistake them for the
``@rpath/libmlx.dylib`` image a Python ``mlx`` loads into the same process.
The binaries are otherwise unmodified, except for the ad-hoc re-signature
that ``install_name_tool`` makes necessary.

A staged entry is never modified in place: it is built in a temporary
directory and renamed into position, so parallel builds (several worktrees
at once) either see a complete entry or none.  Every checkout's engine embeds
the absolute path of the entry as an rpath, so the entry is shared per
machine rather than kept per checkout.

Environment
-----------
LUCID_MLX_CACHE
    Cache root holding the downloaded wheel (``<sha256>.whl``) and the staged
    entries.  Default ``~/Library/Caches/lucid/mlx-metal``.  Deleting an
    entry breaks every engine built against it until the next build (or a
    plain ``python3 tools/fetch_mlx.py``) stages it again.
LUCID_MLX_DIST_DIR
    Offline builds: an already-unpacked ``mlx-metal`` wheel (a directory with
    ``mlx/include`` + ``mlx/lib``, or ``include`` + ``lib``).  Nothing is
    downloaded; the entry is keyed by the hash of its ``libmlx.dylib``.

Usage
-----
``python3 tools/fetch_mlx.py``                 stage and describe the entry
``python3 tools/fetch_mlx.py --print-dir``     stage, print only the entry path
``python3 tools/fetch_mlx.py --print-version`` print only the pinned version
``python3 tools/fetch_mlx.py --bump 0.33.0``   re-pin to another release
                                               (orchestrator only: it is an
                                               engine ABI change)

Standard library only: an sdist build runs this before anything else exists.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import BinaryIO

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
STAMP = ".lucid-staged"
MLX_LIB = "liblucid_mlx.dylib"
JACCL_LIB = "liblucid_jaccl.dylib"
METALLIB = "mlx.metallib"
PYPI_JSON = "https://pypi.org/pypi/mlx-metal/{version}/json"
TIMEOUT = 60  # seconds without progress before a download is abandoned


class FetchError(RuntimeError):
    """Raised when the pinned MLX cannot be obtained or staged."""


@dataclass(frozen=True)
class Pin:
    version: str
    wheel: str
    sha256: str
    url: str
    macos: str


def read_pin(pyproject: Path = PYPROJECT) -> Pin:
    with pyproject.open("rb") as fh:
        table = tomllib.load(fh).get("tool", {}).get("lucid", {}).get("mlx")
    if not isinstance(table, dict):
        raise FetchError(f"{pyproject}: no [tool.lucid.mlx] table")
    fields = ("version", "wheel", "sha256", "url", "macos")
    missing = [key for key in fields if not isinstance(table.get(key), str)]
    if missing:
        raise FetchError(f"{pyproject}: [tool.lucid.mlx] lacks {', '.join(missing)}")
    return Pin(*(str(table[key]) for key in fields))


def cache_root() -> Path:
    override = os.environ.get("LUCID_MLX_CACHE")
    root = Path(override) if override else Path.home() / "Library/Caches/lucid/mlx-metal"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _log(message: str) -> None:
    print(f"fetch_mlx: {message}", file=sys.stderr)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(*command: str) -> str:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise FetchError(
            f"{command[0]} not found: install the Xcode command line tools "
            "(xcode-select --install)"
        ) from exc
    if proc.returncode != 0:
        raise FetchError(f"{' '.join(command)} failed:\n{proc.stderr.strip()}")
    return proc.stdout


def _offline_hint(url: str, exc: OSError) -> FetchError:
    return FetchError(
        f"could not download {url}: {exc}\n"
        "Offline? Point LUCID_MLX_DIST_DIR at an unpacked copy of that wheel."
    )


def _download(url: str, out: BinaryIO) -> None:
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            shutil.copyfileobj(response, out, 1 << 20)
    except OSError as exc:  # URLError, timeouts and resets are all OSError
        raise _offline_hint(url, exc) from exc


def _pypi_wheels(version: str) -> list[tuple[str, str, str]]:
    """``(filename, sha256, url)`` of every file PyPI lists for mlx-metal *version*."""
    url = PYPI_JSON.format(version=version)
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            payload = json.load(response)
    except OSError as exc:
        raise _offline_hint(url, exc) from exc
    try:
        return [
            (str(f["filename"]), str(f["digests"]["sha256"]), str(f["url"]))
            for f in payload["urls"]
        ]
    except (KeyError, TypeError) as exc:
        raise FetchError(f"{url}: unexpected PyPI JSON ({exc!r})") from exc


def fetch_wheel(pin: Pin, cache: Path) -> Path:
    """Return the cached wheel, downloading it first if needed; sha256-checked."""
    wheel = cache / f"{pin.sha256}.whl"
    if wheel.is_file() and _sha256(wheel) == pin.sha256:
        return wheel
    _log(f"downloading {pin.wheel}")
    fd, part_name = tempfile.mkstemp(dir=cache, prefix=".download-", suffix=".whl")
    part = Path(part_name)
    try:
        with os.fdopen(fd, "wb") as out:
            _download(pin.url, out)
        got = _sha256(part)
        if got != pin.sha256:
            raise FetchError(f"{pin.url}: sha256 {got}, pinned {pin.sha256}")
        os.replace(part, wheel)
    finally:
        part.unlink(missing_ok=True)
    return wheel


def _extract(wheel: Path, stage: Path) -> None:
    """Unpack the parts the engine needs from *wheel* into *stage*."""
    renames = {
        "mlx/lib/libmlx.dylib": f"lib/{MLX_LIB}",
        "mlx/lib/libjaccl.dylib": f"lib/{JACCL_LIB}",
        f"mlx/lib/{METALLIB}": f"lib/{METALLIB}",
    }
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            name = info.filename
            if name in renames:
                target = renames.pop(name)
            elif name.startswith("mlx/include/") and not info.is_dir():
                target = "include/" + name.removeprefix("mlx/include/")
            elif name.endswith(".dist-info/licenses/LICENSE"):
                target = "LICENSE"
            else:
                continue
            if ".." in Path(target).parts:
                raise FetchError(f"{wheel}: refusing member {name!r}")
            dest = stage / target
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
    if renames:
        raise FetchError(f"{wheel}: missing {', '.join(sorted(renames))}")


def _extract_pinned(pin: Pin, cache: Path, stage: Path) -> None:
    _extract(fetch_wheel(pin, cache), stage)


_DIST_FILES = ("lib/libmlx.dylib", "lib/libjaccl.dylib", f"lib/{METALLIB}")


def _locate_dist(src: Path) -> Path:
    for root in (src / "mlx", src):
        if (root / "include").is_dir() and all((root / f).is_file() for f in _DIST_FILES):
            return root
    raise FetchError(
        f"LUCID_MLX_DIST_DIR={src}: expected include/ and {', '.join(_DIST_FILES)} "
        "(directly or under mlx/), as in an unpacked mlx-metal wheel"
    )


def _dist_license(root: Path) -> Path:
    """MLX's LICENSE for an unpacked dist; a wheel must not ship without it."""
    candidates = [root / "LICENSE", *sorted(root.parent.glob("*.dist-info/licenses/LICENSE"))]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FetchError(
        f"LUCID_MLX_DIST_DIR: MLX's LICENSE not found (looked for {root / 'LICENSE'} and "
        f"{root.parent}/*.dist-info/licenses/LICENSE); Lucid's wheel must carry it"
    )


def _copy_dist(root: Path, stage: Path) -> None:
    """Copy an unpacked dist into *stage* with the same layout as ``_extract``."""
    shutil.copytree(root / "include", stage / "include")
    (stage / "lib").mkdir()
    shutil.copyfile(root / "lib/libmlx.dylib", stage / "lib" / MLX_LIB)
    shutil.copyfile(root / "lib/libjaccl.dylib", stage / "lib" / JACCL_LIB)
    shutil.copyfile(root / "lib" / METALLIB, stage / "lib" / METALLIB)
    shutil.copyfile(_dist_license(root), stage / "LICENSE")


def _relink(lib: Path) -> None:
    """Give the libraries Lucid's install names, then re-sign them."""
    mlx, jaccl = lib / MLX_LIB, lib / JACCL_LIB
    _run("install_name_tool", "-id", f"@rpath/{JACCL_LIB}", str(jaccl))
    _run(
        "install_name_tool",
        "-id",
        f"@rpath/{MLX_LIB}",
        "-change",
        "@rpath/libjaccl.dylib",
        f"@loader_path/{JACCL_LIB}",
        str(mlx),
    )
    for dylib in (mlx, jaccl):
        _run("codesign", "--force", "--sign", "-", str(dylib))
    linked = _run("otool", "-L", str(mlx))
    if "@rpath/libmlx.dylib" in linked or "@rpath/libjaccl.dylib" in linked:
        raise FetchError(f"{mlx}: install names were not rewritten:\n{linked}")


def _publish(stage: Path, entry: Path, origin: str) -> Path:
    """Atomically move a finished *stage* to *entry*; a concurrent winner is kept."""
    stage.chmod(0o755)  # mkdtemp makes it 0700
    (stage / STAMP).write_text(origin + "\n")
    try:
        stage.rename(entry)
    except OSError:
        shutil.rmtree(stage, ignore_errors=True)
        if not (entry / STAMP).is_file():
            raise
    return entry


def stage(pin: Pin | None = None) -> Path:
    """Return the staged entry for *pin* (default: the repository's pin)."""
    pin = pin or read_pin()
    cache = cache_root()
    dist_dir = os.environ.get("LUCID_MLX_DIST_DIR")
    populate: Callable[[Path], None]
    if dist_dir:
        root = _locate_dist(Path(dist_dir).resolve())
        key = f"dist-{_sha256(root / 'lib/libmlx.dylib')[:12]}"
        origin = f"{pin.version} from {root}"
        populate = partial(_copy_dist, root)
    else:
        key = pin.sha256[:12]
        origin = f"{pin.version} from {pin.wheel} sha256={pin.sha256}"
        populate = partial(_extract_pinned, pin, cache)
    entry = cache / f"mlx-{pin.version}-{key}"
    if (entry / STAMP).is_file():
        return entry
    if entry.exists():
        # Entries appear by rename with the stamp already inside, so a missing
        # stamp is either a lost race (re-check) or damage.
        if (entry / STAMP).is_file():
            return entry
        raise FetchError(f"{entry} exists but is incomplete; delete it and build again")
    work = Path(tempfile.mkdtemp(dir=cache, prefix=".stage-"))
    try:
        populate(work)
        _relink(work / "lib")
        _log(f"staged MLX {pin.version} at {entry}")
        return _publish(work, entry, origin)
    finally:
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)


def _minos(dylib: Path) -> str:
    match = re.search(r"cmd LC_BUILD_VERSION.*?minos (\S+)", _run("otool", "-l", str(dylib)), re.S)
    if match is None:
        raise FetchError(f"{dylib}: no LC_BUILD_VERSION")
    return match.group(1)


def _macos_tag(filename: str) -> tuple[int, int]:
    match = re.search(r"macosx_(\d+)_(\d+)_arm64\.whl$", filename)
    return (int(match.group(1)), int(match.group(2))) if match else (-1, -1)


def _resolve(version: str, floor: str) -> Pin:
    """Look *version* up on PyPI; pick the oldest-macOS arm64 wheel at the *floor*'s major.

    Taking the newest build instead would raise Lucid's macOS floor whenever
    MLX starts publishing for a newer macOS.  The binary's own minos, read
    after staging, is what the pin records.
    """
    major = int(floor.split(".")[0])
    wheels = [w for w in _pypi_wheels(version) if _macos_tag(w[0]) >= (major, 0)]
    if not wheels:
        raise FetchError(f"mlx-metal {version}: no macOS >= {major} arm64 wheel on PyPI")
    filename, sha256, url = min(wheels, key=lambda w: _macos_tag(w[0]))
    return Pin(version, filename, sha256, url, macos=floor)


def _rewrite_pin(pyproject: Path, pin: Pin) -> None:
    text = pyproject.read_text()
    header = text.find("[tool.lucid.mlx]")
    if header < 0:
        raise FetchError(f"{pyproject}: no [tool.lucid.mlx] table")
    end = text.find("\n[", header + 1)
    end = len(text) if end < 0 else end
    section = text[header:end]
    for key in ("version", "wheel", "sha256", "url", "macos"):
        section, count = re.subn(
            rf'^{key} = "[^"]*"$', f'{key} = "{getattr(pin, key)}"', section, flags=re.M
        )
        if count != 1:
            raise FetchError(f"{pyproject}: [tool.lucid.mlx] needs exactly one {key} line")
    pyproject.write_text(text[:header] + section + text[end:])


def bump(version: str) -> Pin:
    """Re-pin to *version*, stage it, and record the macOS floor of its binary."""
    resolved = _resolve(version, read_pin().macos)
    entry = stage(resolved)
    pin = Pin(
        resolved.version,
        resolved.wheel,
        resolved.sha256,
        resolved.url,
        _minos(entry / "lib" / MLX_LIB),
    )
    _rewrite_pin(PYPROJECT, pin)
    return pin


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--print-dir", action="store_true", help="print the staged entry path")
    group.add_argument("--print-version", action="store_true", help="print the pinned version")
    group.add_argument("--bump", metavar="VERSION", help="re-pin to another mlx-metal release")
    args = parser.parse_args(argv)
    try:
        if args.print_version:
            print(read_pin().version)
        elif args.print_dir:
            print(stage())
        elif args.bump:
            pin = bump(args.bump)
            print(f"pinned MLX {pin.version} ({pin.wheel}, macOS >= {pin.macos})")
        else:
            pin = read_pin()
            print(f"MLX {pin.version} (macOS >= {pin.macos}): {stage(pin)}")
    except FetchError as exc:
        print(f"fetch_mlx: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
