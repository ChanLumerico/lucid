#!/usr/bin/env bash
# Pre-commit verification gate for Lucid C++ engine.
#
# Runs every time Claude attempts `git commit`:
#   1. Static CI gates  (check_*.py)   — always, pure Python, < 3 s total
#   2. Forward/backward parity smoke   — skipped if engine .so isn't built
#
# Exit 0 → commit proceeds.
# Exit 1 + JSON {"continue":false,...} → commit blocked; Claude must fix first.
#
# Run manually:  bash tools/pre_commit_check.sh

set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

PYTHON="${PYTHON_BIN:-python3}"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'

pass=0; fail=0
failures=()

run_check() {
    local label="$1"; shift
    if "$@" > /tmp/_lucid_check.txt 2>&1; then
        printf "  ${GREEN}✓${RESET}  %s\n" "$label"
        (( pass++ )) || true
    else
        printf "  ${RED}✗${RESET}  %s\n" "$label"
        sed 's/^/      /' /tmp/_lucid_check.txt
        failures+=("$label")
        (( fail++ )) || true
    fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Lucid pre-commit checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Static CI gates ────────────────────────────────────────────────────
printf "\n${YELLOW}[1/2] Static CI gates${RESET}\n"

run_check "Phase 1  foundation"          "$PYTHON" tools/check_phase1.py
run_check "Phase 16 in-place autograd"   "$PYTHON" tools/check_inplace_ops.py
run_check "Phase 15 BNNS coverage"       "$PYTHON" tools/check_bnns_coverage.py
run_check "Phase 9  unified memory"      "$PYTHON" tools/check_unified_mem.py
run_check "Phase 18 Metal escape hatch"  "$PYTHON" tools/check_metal_escape.py
run_check "Phase 12 custom autograd"     "$PYTHON" tools/check_custom_autograd.py
run_check "Phase 19 op fusion"           "$PYTHON" tools/check_fusion.py
run_check "Layer deps"                   "$PYTHON" tools/check_layers.py
run_check "Op API conformance"           "$PYTHON" tools/check_op_api.py
run_check "Op template conformance"      "$PYTHON" tools/check_op_template.py
run_check "Kernel template coverage"     "$PYTHON" tools/check_kernel_template.py
run_check "Storage API compliance"       "$PYTHON" tools/check_storage_api.py
run_check "Registry coverage"            "$PYTHON" tools/check_registry_coverage.py
run_check "Doxygen coverage ≥70%%"       "$PYTHON" tools/check_doxygen.py --threshold 70
run_check "Stub freshness (gen_pyi)"     "$PYTHON" tools/check_stubs.py

# ── 2. Forward/backward parity smoke tests ────────────────────────────────
printf "\n${YELLOW}[2/2] Forward/backward parity (CPU + GPU)${RESET}\n"

ENGINE_SO="$(find "$REPO/lucid/_C" -name 'engine*.so' 2>/dev/null | head -1)"
if [ -z "$ENGINE_SO" ]; then
    printf "  ${YELLOW}⚠${RESET}  engine.so not found — skipping parity tests\n"
    printf "       (build first:  pip install -e .  then re-run)\n"
else
    # ── core arithmetic: forward + backward on CPU and GPU ────────────────
    CORE_OPS="add or sub or mul or div or matmul or relu or sigmoid or tanh"
    run_check "core ops  forward  CPU" \
        "$PYTHON" -m pytest tests/parity/test_parity.py \
            -k "test_forward_CPU  and ($CORE_OPS)" \
            -q --tb=line --no-header -x

    run_check "core ops  backward CPU" \
        "$PYTHON" -m pytest tests/parity/test_parity.py \
            -k "test_backward_CPU and ($CORE_OPS)" \
            -q --tb=line --no-header -x

    run_check "core ops  forward  GPU" \
        "$PYTHON" -m pytest tests/parity/test_parity.py \
            -k "test_forward_GPU  and ($CORE_OPS)" \
            -q --tb=line --no-header -x

    run_check "core ops  backward GPU" \
        "$PYTHON" -m pytest tests/parity/test_parity.py \
            -k "test_backward_GPU and ($CORE_OPS)" \
            -q --tb=line --no-header -x

    # ── in-place ops (Phase 16) ────────────────────────────────────────────
    run_check "in-place  forward  CPU" \
        "$PYTHON" -m pytest tests/parity/test_parity.py \
            -k "test_forward_CPU  and inplace" \
            -q --tb=line --no-header -x

    run_check "in-place  backward CPU" \
        "$PYTHON" -m pytest tests/parity/test_parity.py \
            -k "test_backward_CPU and inplace" \
            -q --tb=line --no-header -x

    # ── convolution + batch_norm (Phase 15) ────────────────────────────────
    NN_OPS="conv2d or conv1d or batch_norm or layer_norm"
    run_check "nn ops    forward  CPU" \
        "$PYTHON" -m pytest tests/parity/test_parity.py \
            -k "test_forward_CPU  and ($NN_OPS)" \
            -q --tb=line --no-header -x

    run_check "nn ops    backward CPU" \
        "$PYTHON" -m pytest tests/parity/test_parity.py \
            -k "test_backward_CPU and ($NN_OPS)" \
            -q --tb=line --no-header -x

    run_check "nn ops    forward  GPU" \
        "$PYTHON" -m pytest tests/parity/test_parity.py \
            -k "test_forward_GPU  and ($NN_OPS)" \
            -q --tb=line --no-header -x

    run_check "nn ops    backward GPU" \
        "$PYTHON" -m pytest tests/parity/test_parity.py \
            -k "test_backward_GPU and ($NN_OPS)" \
            -q --tb=line --no-header -x

    # ── cross-device consistency check ────────────────────────────────────
    run_check "cross-device forward" \
        "$PYTHON" -m pytest tests/parity/test_parity.py \
            -k "test_cross_device_forward  and ($CORE_OPS or $NN_OPS)" \
            -q --tb=line --no-header -x

    run_check "cross-device backward" \
        "$PYTHON" -m pytest tests/parity/test_parity.py \
            -k "test_cross_device_backward and ($CORE_OPS or $NN_OPS)" \
            -q --tb=line --no-header -x

    # ── error handling + AMP ──────────────────────────────────────────────
    run_check "error handling" \
        "$PYTHON" -m pytest tests/parity/test_errors.py \
            -q --tb=line --no-header -x

    run_check "AMP policy" \
        "$PYTHON" -m pytest tests/parity/test_amp.py \
            -q --tb=line --no-header -x
fi

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ "$fail" -gt 0 ]; then
    printf "${RED}FAILED${RESET}  %d check(s) failed, %d passed\n" "$fail" "$pass"
    echo ""
    echo "  Failing:"
    for f in "${failures[@]}"; do
        printf "    • %s\n" "$f"
    done
    echo ""
    echo "  Fix the issues above, then retry the commit."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    # JSON output → PreToolUse hook blocks the commit
    printf '{"continue":false,"stopReason":"Pre-commit checks failed — see output above."}\n'
    exit 1
else
    printf "${GREEN}PASSED${RESET}  All %d checks green\n" "$pass"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 0
fi
