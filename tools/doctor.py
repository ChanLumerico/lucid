"""Diagnose a source checkout even when its native engine cannot import.

Run ``.venv/bin/python -m tools.doctor`` from the checkout. ``--json`` emits
the same checks for CI. This command inspects only; it never installs or builds.
The runtime probe lives in a child process so a native crash is reported too.
"""

import argparse
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import TypedDict


class Check(TypedDict):
    name: str
    status: str
    detail: str


ROOT = Path(__file__).resolve().parents[1]
REBUILD = "uv pip install --python .venv/bin/python -e . --no-build-isolation"


def source_abi(root: Path) -> tuple[int, int]:
    """Read both contracts without importing the package they protect."""
    tree = ast.parse((root / "lucid/version.py").read_text())
    expected = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_EXPECTED_ABI"
        and node.value is not None
    )
    header = (root / "lucid/_C/version.h").read_text()
    match = re.search(r"^#define\s+LUCID_ABI_VERSION\s+(\d+)\s*$", header, re.MULTILINE)
    if match is None or not isinstance(expected, int):
        raise ValueError("cannot read the Python and native ABI declarations")
    return expected, int(match[1])


def diagnose(timeout: float = 30.0) -> list[Check]:
    checks: list[Check] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "status": "ok" if ok else "error", "detail": detail})

    record(
        "python",
        sys.version_info[:2] == (3, 14),
        f"{platform.python_version()} at {sys.executable}; requires Python 3.14",
    )
    system, machine = platform.system(), platform.machine()
    release = platform.mac_ver()[0]
    record(
        "platform",
        system == "Darwin" and machine == "arm64" and bool(release)
        and int(release.split(".")[0]) >= 15,
        f"{system} {release} {machine}; requires macOS 15+ arm64",
    )
    venv = ROOT / ".venv"
    record(
        "environment",
        not venv.is_dir() or Path(sys.prefix).resolve() == venv.resolve(),
        f"prefix={sys.prefix}; use {venv / 'bin/python'} for this checkout",
    )
    for name in ("mlx", "mlx-metal", "numpy", "safetensors", "setuptools", "wheel", "pybind11"):
        try:
            dist = importlib.metadata.distribution(name)
            record(name, True, f"{dist.version} at {dist.locate_file('')}")
        except importlib.metadata.PackageNotFoundError:
            record(name, False, f"not installed in {sys.executable}")

    search_path = os.pathsep.join((str(Path(sys.executable).parent), os.environ.get("PATH", "")))
    for name in ("cmake", "ninja", "xcrun", "uv"):
        executable = shutil.which(name, path=search_path)
        record(name, executable is not None, executable or "not found on PATH")

    try:
        expected, native = source_abi(ROOT)
        record("source_abi", expected == native, f"Python expects {expected}; C++ header declares {native}")
    except (OSError, SyntaxError, ValueError, StopIteration) as exc:
        record("source_abi", False, str(exc))

    probe = (
        "import json, lucid; from lucid._C import engine as _C_engine; "
        "x = lucid.ones(2, device='cpu'); "
        "assert x.sum().item() == 2; "
        "print(json.dumps({'version': lucid.__version__, "
        "'abi': _C_engine.ABI_VERSION, 'package': lucid.__file__, "
        "'engine': _C_engine.__file__, 'cpu_smoke': True}))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe], cwd=ROOT,
            capture_output=True, text=True, timeout=timeout,
        )
        detail = result.stdout.strip() if result.returncode == 0 else (
            f"exit={result.returncode}\n{result.stderr.strip()}\nRebuild: {REBUILD}"
        )
        record("runtime", result.returncode == 0, detail)
    except (OSError, subprocess.TimeoutExpired) as exc:
        record("runtime", False, f"{exc}\nRebuild: {REBUILD}")
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable checks")
    parser.add_argument("--timeout", type=float, default=30.0, help="runtime probe timeout in seconds")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    checks = diagnose(args.timeout)
    ok = all(check["status"] == "ok" for check in checks)
    if args.json:
        print(json.dumps({"ok": ok, "checks": checks, "rebuild": REBUILD}, indent=2))
    else:
        for check in checks:
            print(f"[{check['status']}] {check['name']}: {check['detail']}")
        if not ok:
            print(f"\nAfter resolving the checks above, rebuild from {ROOT}:\n  {REBUILD}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
