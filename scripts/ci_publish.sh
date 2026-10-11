#!/usr/bin/env bash
# The publish gate: build the wheel (or reuse the one in $LUCID_DIST_DIR,
# default dist/), check what it contains, then install it into a fresh venv
# that has no `mlx` and smoke it there.  publish.yml and release-testpypi.yml
# run this on the wheel they upload; ci_full.sh's checks part builds one.
#
# The engine ships its own pinned MLX (LCD-292, [tool.lucid.mlx]): no `mlx`
# package is installed anywhere here, and the smoke fails if one is imported.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
DIST="${LUCID_DIST_DIR:-dist}"

# The deployment target is the pinned MLX build's minos; setup.py would raise
# a lower one to it anyway.
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-$("$PYTHON_BIN" tools/fetch_mlx.py --print-macos)}"
# The platform *tag* stays macosx-26.0-arm64.  python.org's Python 3.14 is
# universal2, so without this the wheel would claim x86_64 too.  And 26.0, not
# the 26.2 floor: pip spells every macOS 11+ release X_0, so no tag can say
# 26.2 — lucid._C refuses to import below the floor instead.
export _PYTHON_HOST_PLATFORM="${_PYTHON_HOST_PLATFORM:-macosx-26.0-arm64}"
export ARCHFLAGS="${ARCHFLAGS:--arch arm64}"

WHEEL=$(ls "$DIST"/lucid*.whl 2>/dev/null | sort -V | tail -1 || true)
if [ -n "$WHEEL" ]; then
    echo "==> Reusing existing wheel: $WHEEL"
else
    echo "==> Building wheel (target=$MACOSX_DEPLOYMENT_TARGET, plat=$_PYTHON_HOST_PLATFORM)"
    "$PYTHON_BIN" -m pip wheel . -w "$DIST" --no-deps --no-build-isolation -q
    WHEEL=$(ls "$DIST"/lucid*.whl 2>/dev/null | sort -V | tail -1)
    if [ -z "$WHEEL" ]; then
        echo "ERROR: no wheel found in $DIST after build" >&2; exit 1
    fi
    echo "    Built: $WHEEL"
fi

echo "==> Wheel contents, tag and size"
"$PYTHON_BIN" tools/check_wheel.py "$WHEEL"

echo "==> Installing into a fresh venv (no mlx)"
SMOKE_DIR=$(mktemp -d)
trap 'rm -rf "$SMOKE_DIR"' EXIT
"$PYTHON_BIN" -m venv "$SMOKE_DIR/venv"
VENV_PY="$SMOKE_DIR/venv/bin/python"
"$VENV_PY" -m pip install --disable-pip-version-check -q "$WHEEL"

echo "==> Smoke test"
# Run from $SMOKE_DIR: in the checkout, the source tree's lucid/ would shadow
# the installed wheel (cwd is sys.path[0] for `python -`).
PIN=$("$PYTHON_BIN" tools/fetch_mlx.py --print-version)
ENGINE=$(cd "$SMOKE_DIR" && LUCID_MLX_PIN="$PIN" "$VENV_PY" - <<'EOF'
import ctypes
import importlib.util
import os
import sys

assert importlib.util.find_spec("mlx") is None, "the smoke venv must not have mlx"

import lucid
from lucid._C import engine as _C_engine

pin = os.environ["LUCID_MLX_PIN"]
assert _C_engine.mlx_runtime_version() == pin, (_C_engine.mlx_runtime_version(), pin)

x = lucid.arange(0, 24).reshape(2, 3, 4).permute(2, 0, 1)
assert x.shape == (4, 2, 3)
# The engine's own GPU allocation, as lucid.test._fixtures.devices probes it:
# if Metal is unusable here, the error says why rather than a bare False.
_C_engine.zeros([1], _C_engine.F32, _C_engine.GPU)
a = lucid.arange(6, dtype=lucid.float32).reshape(2, 3).to("metal")
assert (a @ a.mT).tolist() == [[5.0, 14.0], [14.0, 50.0]]

# The MLX that answered is the one inside the wheel, not a build-machine copy.
libc = ctypes.CDLL(None)
libc._dyld_image_count.restype = ctypes.c_uint32
libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
libc._dyld_get_image_name.restype = ctypes.c_char_p
images = [libc._dyld_get_image_name(i).decode() for i in range(libc._dyld_image_count())]
loaded = [os.path.realpath(p) for p in images if p.endswith("/liblucid_mlx.dylib")]
here = os.path.realpath(os.path.dirname(_C_engine.__file__))
assert loaded == [os.path.join(here, "liblucid_mlx.dylib")], loaded

stray = sorted(m for m in sys.modules if m.split(".")[0].startswith("mlx"))
assert not stray, f"importing lucid loaded {stray}"
print(f"    import, CPU op and Metal matmul OK on MLX {pin} ({lucid.__file__})", file=sys.stderr)
print(_C_engine.__file__)
EOF
)

echo "==> Installed engine's link"
"$PYTHON_BIN" tools/check_engine_link.py "$ENGINE" --wheel

echo "==> ci_publish: green"
