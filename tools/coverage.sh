#!/usr/bin/env bash
# Build the C++ engine with LLVM coverage instrumentation, run the core test
# suites, and emit an lcov-compatible report.
#
# Phase 0.5 is report-only by default. Set LUCID_COVERAGE_ENFORCE=1 to turn on
# the planned hard gates once the Phase 7/8 parity and benchmark work has
# raised coverage to the target levels.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ "$#" -eq 0 ]]; then
    PYTEST_TARGETS=(tests/parity/)
else
    PYTEST_TARGETS=("$@")
fi
COVERAGE_DIR="${COVERAGE_DIR:-build/coverage}"
ENFORCE_COVERAGE="${LUCID_COVERAGE_ENFORCE:-0}"
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

"$PYTHON_BIN" - "$COVERAGE_DIR/lucid.lcov" "$ENFORCE_COVERAGE" <<'PY'
import sys
from pathlib import Path

lcov_path = Path(sys.argv[1])
enforce = sys.argv[2] == "1"
thresholds = {
    "lucid/_C/ops/": 80.0,
    "lucid/_C/kernel/": 80.0,
    "lucid/_C/backend/": 70.0,
    "lucid/_C/core/": 90.0,
}
totals = {prefix: [0, 0] for prefix in thresholds}
current = None

for line in lcov_path.read_text().splitlines():
    if line.startswith("SF:"):
        current = line[3:]
    elif line.startswith("LF:") and current:
        total = int(line[3:])
        for prefix in totals:
            if prefix in current:
                totals[prefix][1] += total
    elif line.startswith("LH:") and current:
        hit = int(line[3:])
        for prefix in totals:
            if prefix in current:
                totals[prefix][0] += hit
    elif line == "end_of_record":
        current = None

print("==> Phase 0.5 coverage gate status")
failures = []
for prefix, threshold in thresholds.items():
    hit, total = totals[prefix]
    if total == 0:
        status = "no files matched"
        pct = 0.0
    else:
        pct = hit * 100.0 / total
        status = f"{pct:.2f}% ({hit}/{total})"
    print(f"    {prefix:<18} {status}; target {threshold:.1f}%")
    if enforce and total > 0 and pct < threshold:
        failures.append(f"{prefix} {pct:.2f}% < {threshold:.1f}%")

if enforce and failures:
    print("Coverage enforcement failed:", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    raise SystemExit(1)

if not enforce:
    print("==> Coverage thresholds are report-only; set LUCID_COVERAGE_ENFORCE=1 to fail below target.")
PY
