"""
Lucid — educational deep learning framework.

Reconstruction phase: Python API rebuilt on top of the C++ engine
(`lucid._C.engine`). The legacy numpy/MLX-based implementation lives in
`lucid_legacy/` as a read-only reference during this rebuild.

This `__init__.py` is intentionally minimal at the start of Phase 5; it
will be populated incrementally as each submodule is reconstructed.
"""

# Standard convention used throughout the new lucid Python layer:
#   from lucid._C import engine as _C_engine
#   from lucid._C.engine import linalg as _C_linalg
