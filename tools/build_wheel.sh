#!/usr/bin/env bash
# Build an Apple Silicon wheel for the active Python interpreter.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
DIST_DIR="${DIST_DIR:-dist}"

SYSTEM="$("$PYTHON_BIN" - <<'PY'
import platform

print(platform.system())
PY
)"
MACHINE="$("$PYTHON_BIN" - <<'PY'
import platform

print(platform.machine())
PY
)"

if [[ "$SYSTEM" != "Darwin" || "$MACHINE" != "arm64" ]]; then
    echo "Lucid wheels are macOS arm64 Apple Silicon only (got $SYSTEM/$MACHINE)" >&2
    exit 1
fi

rm -rf build "$DIST_DIR"
mkdir -p "$DIST_DIR"

export ARCHFLAGS="-arch arm64"
export MACOSX_DEPLOYMENT_TARGET="14.0"
export LUCID_BUILD_MODE="${LUCID_BUILD_MODE:-release}"

echo "==> Building wheel for $("$PYTHON_BIN" -V)"
"$PYTHON_BIN" setup.py bdist_wheel --plat-name macosx_14_0_arm64 --dist-dir "$DIST_DIR"

echo "==> Built wheels"
ls -1 "$DIST_DIR"/*.whl
