#!/usr/bin/env bash
# Generate `build/compile_commands.json` from the CMake build graph so clangd,
# clang-tidy, and IDE integrations use the same flags as the extension build.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILD_DIR="${BUILD_DIR:-build/compile_commands}"
LUCID_BUILD_MODE="${LUCID_BUILD_MODE:-release}"
MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
SDKROOT="$(xcrun --show-sdk-path)"

find_cmake() {
    if command -v cmake >/dev/null 2>&1; then
        command -v cmake
        return
    fi
    "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import cmake

print(Path(cmake.CMAKE_BIN_DIR) / "cmake")
PY
}

CMAKE_BIN="$(find_cmake)"
mkdir -p "$BUILD_DIR"

PYTHON_INFO="$("$PYTHON_BIN" - <<'PY'
from pathlib import Path
import sysconfig

import pybind11
import mlx.core as mlx_core

mlx_pkg = Path(mlx_core.__file__).resolve().parent
print(sysconfig.get_paths()["include"])
print(pybind11.get_include())
print(mlx_pkg / "include")
print(mlx_pkg / "lib")
print(sysconfig.get_config_var("EXT_SUFFIX"))
PY
)"

CMAKE_PATHS=()
while IFS= read -r line; do
    CMAKE_PATHS+=("$line")
done <<EOF
$PYTHON_INFO
EOF

echo "==> Configuring CMake compile database"
"$CMAKE_BIN" \
    -S lucid/_C \
    -B "$BUILD_DIR" \
    -G Ninja \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_SYSROOT="$SDKROOT" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET" \
    -DLUCID_BUILD_MODE="$LUCID_BUILD_MODE" \
    -DLUCID_PYTHON_INCLUDE_DIR="${CMAKE_PATHS[0]}" \
    -DLUCID_PYBIND11_INCLUDE_DIR="${CMAKE_PATHS[1]}" \
    -DLUCID_MLX_INCLUDE_DIR="${CMAKE_PATHS[2]}" \
    -DLUCID_MLX_LIBRARY_DIR="${CMAKE_PATHS[3]}" \
    -DLUCID_PYTHON_EXTENSION_SUFFIX="${CMAKE_PATHS[4]}" \
    -DLUCID_EXTENSION_OUTPUT_DIR="$(pwd)/lucid/_C"

cp "$BUILD_DIR/compile_commands.json" build/compile_commands.json
echo "==> Wrote $(pwd)/build/compile_commands.json"
