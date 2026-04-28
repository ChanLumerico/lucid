#!/usr/bin/env bash
# Generate `build/compile_commands.json` so clangd / clang-tidy / IDE
# integrations know the exact compile flags for every TU.
#
# Usage:
#   ./scripts/build_compile_commands.sh
#   # then point clangd at build/compile_commands.json (most editors do this
#   # automatically once the file exists in the workspace root or build/).

set -euo pipefail

cd "$(dirname "$0")/.."

# setuptools doesn't emit compile_commands.json natively; we use bear (homebrew)
# to intercept the compiler invocations during a clean rebuild. If bear isn't
# installed, fall back to a hand-built JSON from the existing build artifacts.

if command -v bear >/dev/null 2>&1; then
    echo "==> Generating compile_commands.json via bear"
    rm -rf build
    mkdir -p build

    # Force a full rebuild so bear sees every compile.
    python3 -m pip uninstall -y lucid-dl >/dev/null 2>&1 || true
    bear --output build/compile_commands.json -- \
        python3 -m pip install -e . --no-build-isolation
    echo "==> Wrote $(pwd)/build/compile_commands.json"

else
    echo "==> Generating compile_commands.json from .extensions/cpp_extensions.json"
    mkdir -p build
    python3 - <<'PY'
import json
import os
import platform
import subprocess
import sysconfig
from pathlib import Path

import pybind11

root = Path.cwd()
if platform.system() != "Darwin" or platform.machine() != "arm64":
    raise SystemExit("Lucid C++ engine supports macOS arm64 only")

cfg = json.loads(Path(".extensions/cpp_extensions.json").read_text())
py_inc = sysconfig.get_paths()["include"]
pybind_inc = pybind11.get_include()
sdkroot = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
mlx_inc = None
try:
    import mlx.core as mlx_core

    mlx_inc = str(Path(mlx_core.__file__).resolve().parent / "include")
except ImportError:
    pass

commands = []
for ext in cfg["extensions"]:
    flags = list(ext.get("extra_compile_args", cfg.get("common", {}).get("extra_compile_args", [])))
    for flag in ["-Wall", "-Wextra", "-Wpedantic", "-Werror", "-Wno-unused-parameter"]:
        if flag not in flags:
            flags.append(flag)
    system_includes = [pybind_inc]
    if ext.get("link_mlx") and mlx_inc:
        system_includes.append(mlx_inc)
    include_flags = ["-I", py_inc]
    for inc in ext.get("include_dirs", []):
        include_flags.extend(["-I", inc])
    for inc in system_includes:
        include_flags.extend(["-isystem", inc])
    for src in ext["sources"]:
        commands.append(
            {
                "directory": str(root),
                "file": src,
                "command": " ".join([
                    "clang++",
                    "-arch",
                    "arm64",
                    "-isysroot",
                    sdkroot,
                    *include_flags,
                    *flags,
                    "-c",
                    src,
                ]),
            }
        )

Path("build/compile_commands.json").write_text(json.dumps(commands, indent=2) + "\n")
print(f"wrote {len(commands)} entries to build/compile_commands.json")
PY
fi
