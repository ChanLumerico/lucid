#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET_DIR="$REPO_ROOT/lucid"

if [ ! -d "$TARGET_DIR" ]; then
  echo "Could not find lucid/ directory at $TARGET_DIR" >&2
  exit 1
fi

CXX="${CXX:-clang++}"
STD="${CPP_STD:-c++20}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/.test/cpp_build}"
MODE="${1:---compile}"

if [ "$MODE" != "--compile" ] && [ "$MODE" != "--syntax-only" ]; then
  echo "Usage: $0 [--compile|--syntax-only]" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"

PY_INCLUDES="$(python3-config --includes 2>/dev/null || true)"
PYBIND_INCLUDES="$(python3 -m pybind11 --includes 2>/dev/null || true)"

CPP_FILE_LIST="$BUILD_DIR/.cpp_files.txt"
find "$TARGET_DIR" -type f -name "*.cpp" | sort > "$CPP_FILE_LIST"

CPP_COUNT="$(wc -l < "$CPP_FILE_LIST" | tr -d '[:space:]')"
if [ "$CPP_COUNT" = "0" ]; then
  echo "No .cpp files found under $TARGET_DIR"
  exit 0
fi

echo "Compiler      : $CXX"
echo "C++ standard  : $STD"
echo "Build dir     : $BUILD_DIR"
echo "Files         : $CPP_COUNT"
echo

COMMON_FLAGS=(
  -std="$STD"
  -Wall
  -Wextra
  -Werror
  -I"$REPO_ROOT"
)

if [ -n "$PY_INCLUDES" ]; then
  # shellcheck disable=SC2206
  PY_INC_ARR=($PY_INCLUDES)
  COMMON_FLAGS+=("${PY_INC_ARR[@]}")
fi
if [ -n "$PYBIND_INCLUDES" ]; then
  # shellcheck disable=SC2206
  PYBIND_INC_ARR=($PYBIND_INCLUDES)
  COMMON_FLAGS+=("${PYBIND_INC_ARR[@]}")
fi

while IFS= read -r cpp; do
  rel="${cpp#$REPO_ROOT/}"
  obj="$BUILD_DIR/${rel%.cpp}.o"
  mkdir -p "$(dirname "$obj")"

  echo "Compiling $rel"
  if [ "$MODE" = "--syntax-only" ]; then
    "$CXX" "${COMMON_FLAGS[@]}" -fsyntax-only "$cpp"
  else
    "$CXX" "${COMMON_FLAGS[@]}" -c "$cpp" -o "$obj"
  fi
done < "$CPP_FILE_LIST"

echo
echo "Done."
