#!/usr/bin/env bash
# tools/check_format.sh — Run clang-format compliance check on C++ sources.
#
# Usage:
#   tools/check_format.sh              # check only (exits non-zero on violation)
#   tools/check_format.sh --format-only  # same as above (alias for pre-commit)
#   tools/check_format.sh --tidy       # also run clang-tidy if available

set -euo pipefail

cd "$(dirname "$0")/.."

TIDY=0
for arg in "$@"; do
  [[ "$arg" == "--tidy" ]] && TIDY=1
done

CLANG_FORMAT="${CLANG_FORMAT:-clang-format}"
CPP_FILES=$(find lucid/_C -name "*.h" -o -name "*.hpp" -o -name "*.cpp" | grep -v "/_C/test/" || true)

if [[ -z "$CPP_FILES" ]]; then
  echo "[check_format] No C++ files found."
  exit 0
fi

FAILED=0
for f in $CPP_FILES; do
  if ! "$CLANG_FORMAT" --dry-run --Werror "$f" 2>/dev/null; then
    echo "[check_format] NEEDS FORMAT: $f"
    FAILED=1
  fi
done

if [[ $FAILED -ne 0 ]]; then
  echo "[check_format] Run: clang-format -i lucid/_C/**/*.{h,hpp,cpp}"
  exit 1
fi

echo "[check_format] all C++ files formatted correctly"

if [[ $TIDY -eq 1 ]]; then
  CLANG_TIDY="${CLANG_TIDY:-clang-tidy}"
  if command -v "$CLANG_TIDY" &>/dev/null; then
    echo "[check_format] Running clang-tidy..."
    BUILD_DB="build/temp.macosx-10.15-universal2-cpython-314/lucid__C_engine/compile_commands.json"
    if [[ -f "$BUILD_DB" ]]; then
      $CLANG_TIDY -p "$BUILD_DB" $CPP_FILES 2>&1 | grep -E "warning:|error:" | head -40 || true
    else
      echo "[check_format] compile_commands.json not found — skipping clang-tidy"
    fi
  else
    echo "[check_format] clang-tidy not found — skipping"
  fi
fi
