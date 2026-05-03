#!/bin/bash
set -e

echo "=== lucid Python layer CI ==="

echo "--- Running pytest (Python layer) ---"
python3 -m pytest tests/python/ -x -q

echo "--- Python CI PASS ---"
