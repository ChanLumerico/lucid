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

elif command -v compdb >/dev/null 2>&1; then
    echo "==> Generating compile_commands.json via compdb (fallback)"
    python3 -m pip install -e . --no-build-isolation 2>&1 | tail -3
    # compdb reads the make-style log if you've kept one; this branch is
    # currently a no-op stub. Prefer installing bear: brew install bear.
    echo "WARN: compdb fallback not implemented; install bear: brew install bear" >&2
    exit 1

else
    echo "ERROR: neither 'bear' nor 'compdb' found." >&2
    echo "       Install bear:  brew install bear" >&2
    exit 1
fi
