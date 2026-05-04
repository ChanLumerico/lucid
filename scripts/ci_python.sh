#!/bin/bash
set -e

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=== lucid Python layer CI ==="

echo "--- Running unit + nn + autograd + linalg tests (no torch, no slow) ---"
"$PYTHON_BIN" -m pytest lucid/test/ \
    --ignore=lucid/test/parity \
    --ignore=lucid/test/integration \
    -x -q

echo "--- Python CI PASS ---"
