#!/usr/bin/env bash
# CI gate: build the engine extension with UBSan (and optionally ASan), then
# run the parity test suite under sanitizers. Catches undefined behavior and,
# on supported Pythons, memory leaks / use-after-free.
#
# Usage:
#   ./scripts/ci_sanitizer.sh                # UBSan only
#   ./scripts/ci_sanitizer.sh asan           # ASan + UBSan (requires non-SIP-protected Python)
#
# macOS SIP caveat:
#   The Apple-shipped framework Python at /Library/Frameworks/Python.framework
#   strips DYLD_INSERT_LIBRARIES from launched processes (System Integrity
#   Protection), which prevents ASan's interceptors from loading. Use
#   Homebrew Python or pyenv for ASan runs:
#       brew install python@3.14
#       /opt/homebrew/bin/python3.14 -m venv .venv-asan
#       source .venv-asan/bin/activate
#       ./scripts/ci_sanitizer.sh asan
#
# UBSan does not have this restriction and works on any Python.

set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-ubsan}"

case "$MODE" in
    ubsan)
        BUILD_MODE="debug-ubsan"
        EXTRA_OPTS="UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1"
        ;;
    asan)
        BUILD_MODE="debug-asan"
        # detect_leaks=0 because Python's deferred GC creates apparent leaks
        # that aren't real (we have an explicit gc-aware leak test elsewhere).
        EXTRA_OPTS="ASAN_OPTIONS=detect_leaks=0:abort_on_error=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1"
        ;;
    *)
        echo "Unknown mode: $MODE (use 'ubsan' or 'asan')" >&2
        exit 1
        ;;
esac

echo "==> Building engine with LUCID_BUILD_MODE=$BUILD_MODE"
LUCID_BUILD_MODE="$BUILD_MODE" python3 -m pip install -e . --no-build-isolation \
    2>&1 | tail -5

echo "==> Running parity suite under $MODE"
env $EXTRA_OPTS python3 -m pytest lucid/test/ -m "not slow" --tb=short -q

echo "==> $MODE: green"
