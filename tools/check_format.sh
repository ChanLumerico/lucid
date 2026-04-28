#!/usr/bin/env bash
# Check C++ formatting and clang-tidy diagnostics for the Lucid engine.

set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-}"
RUN_TIDY=1
if [[ "$MODE" == "--format-only" ]]; then
    RUN_TIDY=0
elif [[ -n "$MODE" && "$MODE" != "--tidy" ]]; then
    echo "usage: tools/check_format.sh [--format-only|--tidy]" >&2
    exit 2
fi

CPP_FILES=()
while IFS= read -r file; do
    CPP_FILES+=("$file")
done < <(find lucid/_C \
    \( -name "*.h" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.cc" \) \
    -print | sort)

CPP_TUS=()
while IFS= read -r file; do
    CPP_TUS+=("$file")
done < <(find lucid/_C \
    \( -name "*.cpp" -o -name "*.cc" \) \
    -print | sort)

find_tool() {
    local name="$1"
    if command -v "$name" >/dev/null 2>&1; then
        command -v "$name"
        return 0
    fi
    if command -v xcrun >/dev/null 2>&1; then
        xcrun --find "$name" 2>/dev/null && return 0
    fi
    if [[ -x "/opt/homebrew/opt/llvm/bin/$name" ]]; then
        echo "/opt/homebrew/opt/llvm/bin/$name"
        return 0
    fi
    return 1
}

CLANG_FORMAT="$(find_tool clang-format || true)"
if [[ -z "$CLANG_FORMAT" ]]; then
    echo "error: clang-format is required" >&2
    exit 127
fi

echo "==> Checking clang-format"
format_failed=0
for file in "${CPP_FILES[@]}"; do
    if ! diff -u "$file" <("$CLANG_FORMAT" "$file") >/dev/null; then
        echo "format differs: $file" >&2
        format_failed=1
    fi
done
if [[ "$format_failed" -ne 0 ]]; then
    echo "Run: clang-format -i \$(find lucid/_C -name '*.h' -o -name '*.cpp')" >&2
    exit 1
fi

if [[ "$RUN_TIDY" -eq 0 ]]; then
    echo "clang-format check passed."
    exit 0
fi

CLANG_TIDY="$(find_tool clang-tidy || true)"
if [[ -z "$CLANG_TIDY" ]]; then
    echo "error: clang-tidy is required" >&2
    exit 127
fi

if [[ ! -f build/compile_commands.json ]]; then
    echo "error: build/compile_commands.json missing" >&2
    echo "Run: ./scripts/build_compile_commands.sh" >&2
    exit 2
fi

echo "==> Running clang-tidy"
"$CLANG_TIDY" -p build --quiet --extra-arg=-Wno-error "${CPP_TUS[@]}"
echo "format and clang-tidy checks passed."
