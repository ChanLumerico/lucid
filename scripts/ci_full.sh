#!/usr/bin/env bash
# Full local CI gate for the C++ engine refactor phases.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTEST_TARGET="${PYTEST_TARGET:-tests/parity/}"

echo "==> Release build"
"$PYTHON_BIN" -m pip install -e . --no-build-isolation

echo "==> Parity tests"
"$PYTHON_BIN" -m pytest "$PYTEST_TARGET" --tb=short -q

echo "==> UBSan build + parity"
./scripts/ci_sanitizer.sh ubsan

echo "==> Layer check"
"$PYTHON_BIN" tools/check_layers.py

echo "==> Compile commands"
./scripts/build_compile_commands.sh

echo "==> Format + clang-tidy"
tools/check_format.sh --tidy

echo "==> ci_full: green"
