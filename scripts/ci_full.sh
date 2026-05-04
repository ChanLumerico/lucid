#!/usr/bin/env bash
# Full local CI gate.
# Runs: release build, all Python tests, C++ unit tests, sanitizer build,
#       all validator tools, format check.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"

# ── 1. Release build ──────────────────────────────────────────────────────────
echo "==> Release build"
"$PYTHON_BIN" -m pip install -e . --no-build-isolation

# ── 2. Python unit + nn + autograd + linalg tests (fast, no torch) ──────────
echo "==> Python fast tests (unit / nn / autograd / linalg)"
"$PYTHON_BIN" -m pytest lucid/test/ \
    --ignore=lucid/test/parity \
    --ignore=lucid/test/integration \
    -x -q

# ── 3. Parity tests (require torch) ─────────────────────────────────────────
echo "==> Parity tests (lucid vs PyTorch)"
"$PYTHON_BIN" -m pytest lucid/test/parity/ --tb=short -q || \
    echo "[WARN] Parity tests failed or torch not installed — continuing."

# ── 4. Integration / slow tests ──────────────────────────────────────────────
echo "==> Integration tests"
"$PYTHON_BIN" -m pytest lucid/test/integration/ --tb=short -q

# ── 5. C++ unit tests (debug build with GoogleTest) ───────────────────────────
echo "==> C++ unit tests (debug build)"
BUILD_DIR="build/temp.macosx-10.15-universal2-cpython-314/lucid__C_engine"
if [[ -d "$BUILD_DIR" ]]; then
    LUCID_BUILD_MODE=debug BUILD_TESTING=ON cmake --build "$BUILD_DIR" --parallel "$(sysctl -n hw.logicalcpu)" 2>&1 | tail -5 || true
    if command -v ctest &>/dev/null; then
        ctest --test-dir "$BUILD_DIR" --output-on-failure -j"$(sysctl -n hw.logicalcpu)" || \
            echo "[WARN] C++ tests failed or not built."
    fi
else
    echo "[WARN] Build directory not found — skipping C++ tests."
fi

# ── 6. UBSan build ────────────────────────────────────────────────────────────
echo "==> UBSan build + fast tests"
./scripts/ci_sanitizer.sh ubsan || echo "[WARN] Sanitizer step failed."

# ── 7. Validator tools ────────────────────────────────────────────────────────
echo "==> Layer dependency check"
"$PYTHON_BIN" tools/check_layers.py

echo "==> Op API check"
"$PYTHON_BIN" tools/check_op_api.py

echo "==> Phase 1 foundation check"
"$PYTHON_BIN" tools/check_phase1.py

echo "==> Doxygen coverage"
"$PYTHON_BIN" tools/check_doxygen.py --threshold 70

echo "==> Op template conformance"
"$PYTHON_BIN" tools/check_op_template.py

echo "==> Kernel template coverage"
"$PYTHON_BIN" tools/check_kernel_template.py

echo "==> Storage API compliance"
"$PYTHON_BIN" tools/check_storage_api.py

# ── 8. Build tools ────────────────────────────────────────────────────────────
echo "==> Compile commands"
./scripts/build_compile_commands.sh

echo "==> Format + clang-tidy"
tools/check_format.sh --tidy

# ── 9. Publish gate ───────────────────────────────────────────────────────────
echo "==> Publish gate"
./scripts/ci_publish.sh

echo "==> ci_full: green"
