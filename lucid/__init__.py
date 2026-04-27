"""
Lucid — educational deep learning framework.

Top-level surface: re-exports the entire op set of the new C++ engine
under flat names (``lucid.zeros``, ``lucid.matmul``, ``lucid.relu`` …)
plus the canonical sub-namespaces (``lucid.linalg``, ``lucid.random``,
``lucid.einops``, ``lucid.nn``, ``lucid.optim``, ``lucid.types``,
``lucid.autograd``).

This file mirrors the legacy lucid surface so user code that imports
``lucid.zeros`` or calls ``lucid.no_grad()`` keeps working without
referencing ``lucid.ops.*`` paths explicitly.
"""

from __future__ import annotations

import math as _math
from typing import Any as _Any

# --- core tensor + helpers --------------------------------------------------
from lucid._tensor import (
    Tensor,
    FloatTensor, DoubleTensor, HalfTensor,
    CharTensor, ShortTensor, IntTensor, LongTensor,
)
from lucid._bridge import impl_of, to_engine_dtype, to_engine_device

# --- ops: flat re-export -----------------------------------------------------
from lucid.ops import *  # noqa: F401,F403
from lucid.ops import linalg, random, einops  # noqa: F401

# --- nn / autograd / optim ---------------------------------------------------
from lucid import nn  # noqa: F401
from lucid import autograd  # noqa: F401
from lucid.autograd import no_grad, enable_grad  # noqa: F401
from lucid import optim  # noqa: F401

# --- types -------------------------------------------------------------------
from lucid import types  # noqa: F401
from lucid.types import (  # noqa: F401
    Int, Int8, Int16, Int32, Int64,
    Float, Float16, Float32, Float64,
    Complex, Complex64,
    Bool,
    Numeric,
)

# Common dtype aliases (PyTorch parity).
Char = Int8
Short = Int16
Long = Int64
Half = Float16
Double = Float64

# Common op-name aliases (PyTorch / NumPy convenience).
mul = multiply  # noqa: F405  (re-exported from lucid.ops.bfunc)
truediv = divide if 'divide' in globals() else div  # noqa: F405
neg = negative if 'negative' in globals() else lambda a: -a  # noqa: F405

# --- constants ---------------------------------------------------------------
pi = _math.pi
inf = _math.inf
newaxis = None


# --- Tensor factory helpers --------------------------------------------------

def tensor(
    data,
    requires_grad: bool = False,
    keep_grad: bool = False,
    dtype: _Any | None = None,
    device: str = "cpu",
) -> "Tensor":
    """Construct a Tensor from data.  Mirrors the legacy ``lucid.tensor``."""
    if isinstance(data, Tensor):
        # Wrap an existing Tensor (no copy when same dtype/device).
        if (dtype is None or data.dtype is dtype) and data.device == device:
            t = data
        else:
            t = data.astype(dtype) if dtype is not None else data
            if t.device != device:
                t = t.to(device)
    else:
        t = Tensor(data, dtype=dtype, device=device)
    if requires_grad:
        t.requires_grad_(True)
    return t


def to_tensor(
    a,
    requires_grad: bool = False,
    keep_grad: bool = False,
    dtype=None,
    device: str = "cpu",
) -> "Tensor":
    return tensor(a, requires_grad, keep_grad, dtype, device)


def shape(a):
    if hasattr(a, "shape"):
        return a.shape
    raise ValueError("The argument must be a Tensor or array-like with .shape.")


# --- legacy-compat aliases ---------------------------------------------------
# Some callers still use ``lucid.no_grad()`` (no parens-less form).  Both
# context-manager and decorator forms are provided by ``autograd.no_grad``
# itself, so this module-level re-export is a verbatim re-export.

# Top-level grad-state queries (read-only)
def grad_enabled() -> bool:
    """True if autograd is currently enabled in this thread."""
    from lucid._C import engine as _eng
    return _eng.GradMode.is_enabled()


# --- Tensor arithmetic + comparison + indexing dunders ----------------------
# Mirrors the legacy lucid binding: every arithmetic / compare op routes
# through the Python-level lucid.ops layer (which itself wraps the C++ engine).
# Scalar arguments are auto-promoted via lucid.tensor() before dispatch.

def _as_tensor(x, like: "Tensor | None" = None):
    if isinstance(x, Tensor):
        if like is not None and x.shape == ():
            # 0-d → broadcast to match like's shape (engine bin-ops still
            # don't broadcast on their own).
            return broadcast_to(x, like.shape) if like.shape else x  # noqa: F405
        return x
    if like is not None:
        # Coerce scalar to match `like`'s dtype/device so engine ops accept,
        # then broadcast 0-d to like's shape (mul/add/etc. need matching).
        t = tensor(x, dtype=like.dtype, device=like.device)
        if like.shape:
            t = broadcast_to(t, like.shape)  # noqa: F405
        return t
    return tensor(x)


def _binop(op):
    def _wrap(self, other):
        return op(self, _as_tensor(other, like=self))
    return _wrap


def _rbinop(op):
    def _wrap(self, other):
        return op(_as_tensor(other, like=self), self)
    return _wrap


# Pull the actual op functions from the flat namespace.
from lucid.ops.bfunc import (
    add as _add,
    sub as _sub,
    multiply as _mul,
    div as _div,
    _floordiv as _floordiv_op,
    matmul as _matmul,
    power as _pow_op,
)
from lucid.ops.ufunc import _neg as _neg_op, _invert as _invert_op
from lucid.ops.bfunc import _bitwise_and, _bitwise_or

Tensor.__add__       = _binop(_add)
Tensor.__radd__      = _rbinop(_add)
Tensor.__sub__       = _binop(_sub)
Tensor.__rsub__      = _rbinop(_sub)
Tensor.__mul__       = _binop(_mul)
Tensor.__rmul__      = _rbinop(_mul)
Tensor.__truediv__   = _binop(_div)
Tensor.__rtruediv__  = _rbinop(_div)
Tensor.__floordiv__  = _binop(_floordiv_op)
Tensor.__rfloordiv__ = _rbinop(_floordiv_op)
Tensor.__matmul__    = _binop(_matmul)
Tensor.__pow__       = _binop(_pow_op)
Tensor.__rpow__      = _rbinop(_pow_op)
Tensor.__neg__       = lambda self: _neg_op(self)
Tensor.__invert__    = lambda self: _invert_op(self)


# Comparison dunders. These return Tensor (Bool) — needed for masks.
from lucid.ops.bfunc import (
    _equal, _not_equal,
    _greater, _greater_or_equal,
    _less, _less_or_equal,
)
Tensor.__eq__ = _binop(_equal)
Tensor.__ne__ = _binop(_not_equal)
Tensor.__gt__ = _binop(_greater)
Tensor.__ge__ = _binop(_greater_or_equal)
Tensor.__lt__ = _binop(_less)
Tensor.__le__ = _binop(_less_or_equal)
# Hash must remain identity (Tensor is mutable).
Tensor.__hash__ = lambda self: id(self)


# Bitwise (mostly used on Bool tensors).
Tensor.__and__  = _binop(_bitwise_and)
Tensor.__rand__ = _rbinop(_bitwise_and)
Tensor.__or__   = _binop(_bitwise_or)
Tensor.__ror__  = _rbinop(_bitwise_or)


# Lazy attr: keeps import side-effects minimal for rarely-used surfaces.
def __getattr__(name: str) -> _Any:
    # Lazy resolution of optional submodules (visual / data / weights).
    if name == "visual":
        try:
            from lucid import visual as _v
        except ImportError as e:  # pragma: no cover
            raise AttributeError(name) from e
        return _v
    if name == "register_model":
        # Pulled in lazily to avoid pulling in models registry on every import.
        from lucid.models import register_model as _r
        return _r
    raise AttributeError(name)
