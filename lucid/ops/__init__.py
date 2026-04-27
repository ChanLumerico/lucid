"""
lucid.ops — Python-side op surface over the C++ engine.

Layout:
  * Flat-imported (re-exported at `lucid.ops.<name>` and at top-level
    `lucid.<name>`):
      - bfunc:  binary ops (add, sub, matmul, ...)
      - ufunc:  unary ops + reductions (exp, log, sum, mean, ...)
      - gfunc:  tensor generation (zeros, ones, arange, ...)
      - utils:  shape / indexing utilities (reshape, stack, ...)

  * Sub-namespaces (re-exported at top-level as `lucid.linalg`, etc.):
      - linalg:  inv, det, solve, qr, svd, ...
      - random:  rand, randn, manual_seed, Generator, ...
      - einops:  rearrange, reduce, repeat
"""

from __future__ import annotations

from lucid.ops import linalg, random, einops

from lucid.ops.bfunc import *
from lucid.ops.ufunc import *
from lucid.ops.gfunc import *
from lucid.ops.utils import *

from lucid.ops.bfunc import __all__ as _bfunc_all
from lucid.ops.ufunc import __all__ as _ufunc_all
from lucid.ops.gfunc import __all__ as _gfunc_all
from lucid.ops.utils import __all__ as _utils_all


__all__ = (
    list(_bfunc_all)
    + list(_ufunc_all)
    + list(_gfunc_all)
    + list(_utils_all)
    + ["linalg", "random", "einops"]
)
