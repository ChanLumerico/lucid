#!/usr/bin/env bash
set -euo pipefail

clean_build=false
skip_make=false

for arg in "$@"; do
  case "$arg" in
    --clean) clean_build=true ;;
    --skip-make) skip_make=true ;;
    -h|--help)
      cat <<'EOF'
Usage: build_html.sh [--clean] [--skip-make]
  --clean      Remove docs/build/ before building.
  --skip-make  Skip "make html".
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

cd devtools/
./black.sh ../lucid
./compile.sh
./clean_pycache.sh

python --version
cd ../docs
if $clean_build; then
  rm -rf build/
fi
if ! $skip_make; then
  make html
fi
