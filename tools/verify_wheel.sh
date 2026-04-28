#!/usr/bin/env bash
# Install the built wheel in a fresh virtualenv and verify the C++ extension
# imports from that wheel.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
DIST_DIR="${DIST_DIR:-dist}"
VENV_DIR="${VENV_DIR:-build/wheel-verify-venv}"
REPO_ROOT="$(pwd)"

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

WHEELS=()
while IFS= read -r wheel; do
    WHEELS+=("$wheel")
done < <(find "$DIST_DIR" -maxdepth 1 -name '*.whl' -type f | sort)
if [[ "${#WHEELS[@]}" -ne 1 ]]; then
    echo "Expected exactly one wheel in $DIST_DIR, found ${#WHEELS[@]}" >&2
    printf '  %s\n' "${WHEELS[@]:-}" >&2
    exit 1
fi

rm -rf "$VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install --force-reinstall "${WHEELS[0]}"

VERIFY_CWD="$(mktemp -d)"
trap 'rm -rf "$VERIFY_CWD"' EXIT

(
cd "$VERIFY_CWD"
"$REPO_ROOT/$VENV_DIR/bin/python" - "$REPO_ROOT/$VENV_DIR" <<'PY'
import platform
import sys
from pathlib import Path

import lucid._C.engine as engine

venv_dir = Path(sys.argv[1]).resolve()
engine_path = Path(engine.__file__).resolve()
if venv_dir not in engine_path.parents:
    raise SystemExit(f"engine imported outside fresh venv: {engine_path}")

print(f"runner={platform.system()}/{platform.machine()}")
print(f"engine={engine_path}")
PY
)

echo "==> Wheel import verified in fresh venv: ${WHEELS[0]}"
