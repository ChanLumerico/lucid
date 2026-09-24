#!/usr/bin/env bash
# C++ unit tests (GoogleTest, the ``lucid_test`` target under lucid/_C/test).
#
# The gate used to look for the build tree setup.py leaves behind, under a
# path pip no longer uses, and printed "[WARN] Build directory not found —
# skipping C++ tests" on every run: the 147 tests never ran in CI.  This
# configures its own tree the way build_compile_commands.sh does, with
# BUILD_TESTING on and the extension written inside the tree rather than
# over the installed one, builds the test binary and runs it.  A failure
# fails the gate.
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_DIR="${BUILD_DIR:-build/cpp_tests}"
JOBS="$(sysctl -n hw.logicalcpu)"

BUILD_DIR="$BUILD_DIR" \
    EXTRA_CMAKE_ARGS="-DBUILD_TESTING=ON -DLUCID_EXTENSION_OUTPUT_DIR=$(pwd)/$BUILD_DIR/ext" \
    ./scripts/build_compile_commands.sh
cmake --build "$BUILD_DIR" --target lucid_test --parallel "$JOBS"
ctest --test-dir "$BUILD_DIR" --output-on-failure -j"$JOBS"
