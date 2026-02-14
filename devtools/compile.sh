#!/usr/bin/env bash
set -euo pipefail

MODE="inplace"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MACOS_TARGET="${MACOSX_DEPLOYMENT_TARGET:-10.15}"

for arg in "$@"; do
  case "$arg" in
    --inplace) MODE="inplace" ;;
    --dist) MODE="dist" ;;
    -h|--help)
      cat <<'EOF'
Usage: build_setup.sh [--inplace|--dist]
  --inplace  Run: python setup.py build_ext --inplace (default)
  --dist     Run: python setup.py sdist bdist_wheel

Environment:
  PYTHON_BIN                  Python executable to use (default: python)
  MACOSX_DEPLOYMENT_TARGET    macOS deployment target (default: 10.15)
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_CONFIG_PATH="$REPO_ROOT/.extensions/cpp_extensions.json"

cd "$REPO_ROOT"
export MACOSX_DEPLOYMENT_TARGET="$MACOS_TARGET"

echo "Python: $PYTHON_BIN"
echo "Python executable: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET"

if ! "$PYTHON_BIN" -c "import pybind11" >/dev/null 2>&1; then
  echo "pybind11 is not installed for $PYTHON_BIN." >&2
  echo "Install with: $PYTHON_BIN -m pip install pybind11" >&2
  exit 1
fi

if [ "$MODE" = "inplace" ]; then
  "$PYTHON_BIN" setup.py build_ext --inplace
else
  "$PYTHON_BIN" setup.py sdist bdist_wheel
fi

if [ "$MODE" = "inplace" ]; then
  if "$PYTHON_BIN" -m pybind11_stubgen --help >/dev/null 2>&1; then
    MODULES_STR="$(
      "$PYTHON_BIN" - <<PY
import json
from pathlib import Path

cfg_path = Path(r"$BUILD_CONFIG_PATH")
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
mods = cfg.get("stub_modules", [])
for m in mods:
    print(m)
PY
    )"
    if [ -z "$MODULES_STR" ]; then
      echo "[stubgen] no modules in $BUILD_CONFIG_PATH (stub_modules)." >&2
      exit 1
    fi
    while IFS= read -r mod; do
      [ -z "$mod" ] && continue
      echo "[stubgen] $mod"
      if ! "$PYTHON_BIN" -m pybind11_stubgen "$mod" -o .; then
        echo "[stubgen] warning: failed for $mod; keeping existing .pyi files (if any)." >&2
        break
      fi
    done <<EOF
$MODULES_STR
EOF
  else
    echo "[stubgen] pybind11-stubgen not installed for $PYTHON_BIN; skipping." >&2
  fi
fi

cd devtools/
./black.sh ../lucid/
