#!/usr/bin/env bash
# CI gate: build a distribution wheel and verify it installs and imports correctly.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==> Building wheel"
"$PYTHON_BIN" -m pip wheel . -w dist/ --no-deps --no-build-isolation -q

WHEEL=$(ls dist/lucid*.whl 2>/dev/null | sort -V | tail -1)
if [ -z "$WHEEL" ]; then
    echo "ERROR: no wheel found in dist/" >&2; exit 1
fi
echo "    Built: $WHEEL"

echo "==> Installing wheel into temp venv"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT
"$PYTHON_BIN" -m venv "$TMPDIR/venv"
source "$TMPDIR/venv/bin/activate"

python -m pip install "$WHEEL" -q --no-deps 2>&1 | tail -3

echo "==> Smoke test"
python - <<'EOF'
from lucid._C import engine as E
t = E.randn([3, 4])
assert t.shape == [3, 4], f"shape mismatch: {t.shape}"
assert str(t.dtype) in ("Dtype.F32", "F32")
print("    engine import OK, tensor creation OK")
EOF

echo "==> ci_publish: green"
