#!/usr/bin/env bash
# CI gate: build a distribution wheel and verify it installs and imports correctly.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Force the wheel platform tag to macosx_26_0_arm64. python.org's Python
# 3.14 is a universal2 interpreter and would otherwise yield a
# `macosx_26_0_universal2` tag advertising x86_64 compatibility for an
# arm64-only engine. CI workflows set these too — duplicated here so
# local `./scripts/ci_publish.sh` invocations produce the same artifact.
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-26.0}"
export _PYTHON_HOST_PLATFORM="${_PYTHON_HOST_PLATFORM:-macosx-26.0-arm64}"
export ARCHFLAGS="${ARCHFLAGS:--arch arm64}"

echo "==> Building wheel (target=$MACOSX_DEPLOYMENT_TARGET, plat=$_PYTHON_HOST_PLATFORM)"
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
# CRITICAL: `cd` out of the project root before importing. The repo's
# `lucid/` source directory shadows the installed wheel in `sys.path`
# (cwd is `sys.path[0]` for `python - <<EOF`). The source tree never
# contains the compiled `engine.cpython-*-darwin.so` — only the wheel
# does — so an in-tree import surfaces a misleading circular-import
# error from `lucid/_C/__init__.py`. Running from `$TMPDIR` (which has
# no `lucid/`) lets Python resolve through the venv's site-packages.
( cd "$TMPDIR" && python - <<'EOF'
import lucid
from lucid._C import engine as E
t = E.randn([3, 4])
assert t.shape == [3, 4], f"shape mismatch: {t.shape}"
assert str(t.dtype) in ("Dtype.F32", "F32")
print(f"    engine import OK ({lucid.__file__}), tensor creation OK")
EOF
)

echo "==> ci_publish: green"
