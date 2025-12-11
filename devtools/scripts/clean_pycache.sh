#!/bin/bash
set -euo pipefail

# Remove every __pycache__ directory under the repository's lucid/ package.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET_DIR="$REPO_ROOT/lucid"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Could not find lucid/ directory at $TARGET_DIR"
    exit 1
fi

echo "Removing __pycache__ directories under $TARGET_DIR ..."
find "$TARGET_DIR" -type d -name "__pycache__" -exec rm -rf {} +
echo "Done."
