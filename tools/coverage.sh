#!/usr/bin/env bash
# Build the C++ engine with LLVM coverage instrumentation, run the core test
# suites, and emit an lcov-compatible report.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ "$#" -eq 0 ]]; then
    PYTEST_TARGETS=(tests/parity/)
else
    PYTEST_TARGETS=("$@")
fi
COVERAGE_DIR="${COVERAGE_DIR:-build/coverage}"
mkdir -p "$COVERAGE_DIR"
rm -f "$COVERAGE_DIR"/*.profraw "$COVERAGE_DIR"/*.profdata "$COVERAGE_DIR"/*.lcov

echo "==> Building coverage-instrumented engine"
LUCID_BUILD_MODE=debug \
LUCID_COVERAGE=ON \
"$PYTHON_BIN" -m pip install -e . --no-build-isolation

echo "==> Running coverage targets"
LLVM_PROFILE_FILE="$COVERAGE_DIR/lucid-%p.profraw" \
"$PYTHON_BIN" -m pytest "${PYTEST_TARGETS[@]}" --tb=short -q

ENGINE_SO="$("$PYTHON_BIN" - <<'PY'
import lucid._C.engine as engine

print(engine.__file__)
PY
)"

echo "==> Merging raw profiles"
xcrun llvm-profdata merge -sparse "$COVERAGE_DIR"/*.profraw -o "$COVERAGE_DIR/lucid.profdata"

echo "==> Writing lcov report"
xcrun llvm-cov export \
    "$ENGINE_SO" \
    -instr-profile="$COVERAGE_DIR/lucid.profdata" \
    -format=lcov \
    -ignore-filename-regex='(/Library/|/Applications/|site-packages|pybind11)' \
    > "$COVERAGE_DIR/lucid.lcov"

echo "==> Coverage summary"
xcrun llvm-cov report \
    "$ENGINE_SO" \
    -instr-profile="$COVERAGE_DIR/lucid.profdata" \
    -ignore-filename-regex='(/Library/|/Applications/|site-packages|pybind11)'

echo "==> Wrote $COVERAGE_DIR/lucid.lcov"
