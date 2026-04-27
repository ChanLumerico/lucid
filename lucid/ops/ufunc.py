"""
lucid.ops.ufunc — unary operations (mirrors `lucid_legacy/_func/ufunc.py`).

Includes: arith (neg/abs/sign/.../square/cube), exp/log/sqrt, trig
(sin/cos/.../arctan), hyperbolic (sinh/cosh/tanh), activations
(relu/sigmoid/silu/gelu/leaky_relu/softplus/softmax), scalar-param
(clip, _pow, _rpow), discrete (round/floor/ceil/_invert), reductions
(sum/mean/max/min/prod/var/trace/cumsum/cumprod), and transpose family
(transpose/swapaxes/_T/_mT). Plus all in-place variants.
"""

from __future__ import annotations

from typing import Callable

from lucid._C import engine as _C_engine
from lucid._tensor import Tensor
from lucid._bridge import impl_of


__all__ = [
    # Arith
    "_neg", "_invert", "abs", "sign", "reciprocal", "square", "cube",
    # Exp / log
    "exp", "log", "log2", "sqrt",
    # Trig
    "sin", "cos", "tan", "arcsin", "arccos", "arctan",
    # Hyperbolic
    "sinh", "cosh", "tanh",
    # Activation
    "relu", "sigmoid", "silu", "gelu", "leaky_relu", "softplus", "softmax",
    "elu", "selu", "mish", "hard_sigmoid", "hard_swish", "relu6",
    # Scalar-param
    "_pow", "_rpow", "clip",
    # Discrete
    "round", "floor", "ceil",
    # Transpose family
    "_T", "_mT", "transpose", "swapaxes",
    # Reductions
    "sum", "trace", "mean", "var", "min", "max",
    # Scans
    "cumsum", "cumprod",
    # In-place
    "neg_", "abs_", "sign_", "reciprocal_", "square_", "cube_",
    "exp_", "log_", "log2_", "sqrt_",
    "sin_", "cos_", "tan_", "arcsin_", "arccos_", "arctan_",
    "sinh_", "cosh_", "tanh_",
    "round_", "floor_", "ceil_", "clip_",
]


def _wrap1(eng_fn: Callable, a: Tensor) -> Tensor:
    return Tensor._wrap(eng_fn(impl_of(a)))


# --------------------------------------------------------------------------- #
# Arith
# --------------------------------------------------------------------------- #

def _neg(a: Tensor, /) -> Tensor:        return _wrap1(_C_engine.neg, a)
def _invert(a: Tensor, /) -> Tensor:     return _wrap1(_C_engine.invert, a)
def abs(a: Tensor, /) -> Tensor:         return _wrap1(_C_engine.abs, a)
def sign(a: Tensor, /) -> Tensor:        return _wrap1(_C_engine.sign, a)
def reciprocal(a: Tensor, /) -> Tensor:  return _wrap1(_C_engine.reciprocal, a)
def square(a: Tensor, /) -> Tensor:      return _wrap1(_C_engine.square, a)
def cube(a: Tensor, /) -> Tensor:        return _wrap1(_C_engine.cube, a)


# --------------------------------------------------------------------------- #
# Exp / log
# --------------------------------------------------------------------------- #

def exp(a: Tensor, /) -> Tensor:    return _wrap1(_C_engine.exp, a)
def log(a: Tensor, /) -> Tensor:    return _wrap1(_C_engine.log, a)
def log2(a: Tensor, /) -> Tensor:   return _wrap1(_C_engine.log2, a)
def sqrt(a: Tensor, /) -> Tensor:   return _wrap1(_C_engine.sqrt, a)


# --------------------------------------------------------------------------- #
# Trig / Hyperbolic
# --------------------------------------------------------------------------- #

def sin(a: Tensor, /) -> Tensor:    return _wrap1(_C_engine.sin, a)
def cos(a: Tensor, /) -> Tensor:    return _wrap1(_C_engine.cos, a)
def tan(a: Tensor, /) -> Tensor:    return _wrap1(_C_engine.tan, a)
def arcsin(a: Tensor, /) -> Tensor: return _wrap1(_C_engine.arcsin, a)
def arccos(a: Tensor, /) -> Tensor: return _wrap1(_C_engine.arccos, a)
def arctan(a: Tensor, /) -> Tensor: return _wrap1(_C_engine.arctan, a)
def sinh(a: Tensor, /) -> Tensor:   return _wrap1(_C_engine.sinh, a)
def cosh(a: Tensor, /) -> Tensor:   return _wrap1(_C_engine.cosh, a)
def tanh(a: Tensor, /) -> Tensor:   return _wrap1(_C_engine.tanh, a)


# --------------------------------------------------------------------------- #
# Activations
# --------------------------------------------------------------------------- #

def relu(a: Tensor, /) -> Tensor:    return _wrap1(_C_engine.relu, a)
def sigmoid(a: Tensor, /) -> Tensor: return _wrap1(_C_engine.sigmoid, a)
def silu(a: Tensor, /) -> Tensor:    return _wrap1(_C_engine.silu, a)
def gelu(a: Tensor, /) -> Tensor:    return _wrap1(_C_engine.gelu, a)
def leaky_relu(a: Tensor, /, slope: float = 0.01) -> Tensor:
    return Tensor._wrap(_C_engine.leaky_relu(impl_of(a), float(slope)))
def softplus(a: Tensor, /) -> Tensor:  return _wrap1(_C_engine.softplus, a)
def softmax(a: Tensor, /, axis: int = -1) -> Tensor:
    return Tensor._wrap(_C_engine.softmax(impl_of(a), int(axis)))
def elu(a: Tensor, /, alpha: float = 1.0) -> Tensor:
    return Tensor._wrap(_C_engine.elu(impl_of(a), float(alpha)))
def selu(a: Tensor, /) -> Tensor:         return _wrap1(_C_engine.selu, a)
def mish(a: Tensor, /) -> Tensor:         return _wrap1(_C_engine.mish, a)
def hard_sigmoid(a: Tensor, /) -> Tensor: return _wrap1(_C_engine.hard_sigmoid, a)
def hard_swish(a: Tensor, /) -> Tensor:   return _wrap1(_C_engine.hard_swish, a)
def relu6(a: Tensor, /) -> Tensor:        return _wrap1(_C_engine.relu6, a)


# --------------------------------------------------------------------------- #
# Scalar-parameterized
# --------------------------------------------------------------------------- #

def _pow(a: Tensor, /, exp: float) -> Tensor:
    return Tensor._wrap(_C_engine.pow_scalar(impl_of(a), float(exp)))


def _rpow(a: Tensor, /, base: float) -> Tensor:
    return Tensor._wrap(_C_engine.rpow_scalar(float(base), impl_of(a)))


def clip(a: Tensor, /, min_value=None, max_value=None) -> Tensor:
    if min_value is None:
        # Use min(a) of the tensor; matches legacy semantics.
        from lucid.ops.ufunc import min as _min_op
        min_value = _min_op(a).item()
    if max_value is None:
        from lucid.ops.ufunc import max as _max_op
        max_value = _max_op(a).item()
    return Tensor._wrap(_C_engine.clip(impl_of(a), float(min_value),
                                       float(max_value)))


# --------------------------------------------------------------------------- #
# Discrete (no grad)
# --------------------------------------------------------------------------- #

def round(a: Tensor, /, decimals: int = 0) -> Tensor:
    if decimals != 0:
        # Engine round op currently ignores decimals; emulate via scaling.
        from lucid.ops.bfunc import multiply, div
        scale = 10.0 ** decimals
        scale_t = Tensor([[scale]] if a.ndim >= 1 else scale)
        # Simpler: do via numpy-side fast-path to avoid extra ops.
        import numpy as np
        out = np.round(a.numpy(), decimals=decimals)
        return Tensor(out, dtype=a.dtype, device=a.device)
    return _wrap1(_C_engine.round, a)


def floor(a: Tensor) -> Tensor:  return _wrap1(_C_engine.floor, a)
def ceil(a: Tensor) -> Tensor:   return _wrap1(_C_engine.ceil, a)


# --------------------------------------------------------------------------- #
# Transpose family
# --------------------------------------------------------------------------- #

def _T(a: Tensor, /) -> Tensor:    return _wrap1(_C_engine.T, a)
def _mT(a: Tensor, /) -> Tensor:   return _wrap1(_C_engine.mT, a)


def transpose(a: Tensor, /, axes: list[int] | None = None) -> Tensor:
    if axes is None:
        return _wrap1(_C_engine.transpose, a)
    return Tensor._wrap(_C_engine.permute(impl_of(a), list(axes)))


def swapaxes(a: Tensor, /, axis1: int, axis2: int) -> Tensor:
    return Tensor._wrap(_C_engine.swapaxes(impl_of(a), int(axis1), int(axis2)))


# --------------------------------------------------------------------------- #
# Reductions
# --------------------------------------------------------------------------- #

def _normalize_axes(axis):
    """axis: None | int | tuple/list[int] → list[int] for the engine."""
    if axis is None:
        return []
    if isinstance(axis, int):
        return [axis]
    return [int(a) for a in axis]


def sum(a: Tensor, /, axis=None, keepdims: bool = False) -> Tensor:
    return Tensor._wrap(_C_engine.sum(impl_of(a), _normalize_axes(axis), bool(keepdims)))


def mean(a: Tensor, /, axis=None, keepdims: bool = False) -> Tensor:
    return Tensor._wrap(_C_engine.mean(impl_of(a), _normalize_axes(axis), bool(keepdims)))


def var(a: Tensor, /, axis=None, keepdims: bool = False) -> Tensor:
    return Tensor._wrap(_C_engine.var(impl_of(a), _normalize_axes(axis), bool(keepdims)))


def min(a: Tensor, /, axis=None, keepdims: bool = False) -> Tensor:
    return Tensor._wrap(_C_engine.min(impl_of(a), _normalize_axes(axis), bool(keepdims)))


def max(a: Tensor, /, axis=None, keepdims: bool = False) -> Tensor:
    return Tensor._wrap(_C_engine.max(impl_of(a), _normalize_axes(axis), bool(keepdims)))


def trace(a: Tensor, /) -> Tensor:
    return _wrap1(_C_engine.trace, a)


def cumsum(a: Tensor, axis: int = -1) -> Tensor:
    return Tensor._wrap(_C_engine.cumsum(impl_of(a), int(axis)))


def cumprod(a: Tensor, axis: int = -1) -> Tensor:
    return Tensor._wrap(_C_engine.cumprod(impl_of(a), int(axis)))


# --------------------------------------------------------------------------- #
# In-place variants — mutate `a`'s storage and bump version
# --------------------------------------------------------------------------- #

def _inplace1(eng_fn: Callable, a: Tensor) -> Tensor:
    eng_fn(impl_of(a))
    return a


def neg_(a: Tensor, /) -> Tensor:        return _inplace1(_C_engine.neg_, a)
def abs_(a: Tensor, /) -> Tensor:        return _inplace1(_C_engine.abs_, a)
def sign_(a: Tensor, /) -> Tensor:       return _inplace1(_C_engine.sign_, a)
def reciprocal_(a: Tensor, /) -> Tensor: return _inplace1(_C_engine.reciprocal_, a)
def square_(a: Tensor, /) -> Tensor:     return _inplace1(_C_engine.square_, a)
def cube_(a: Tensor, /) -> Tensor:       return _inplace1(_C_engine.cube_, a)
def exp_(a: Tensor, /) -> Tensor:        return _inplace1(_C_engine.exp_, a)
def log_(a: Tensor, /) -> Tensor:        return _inplace1(_C_engine.log_, a)
def log2_(a: Tensor, /) -> Tensor:       return _inplace1(_C_engine.log2_, a)
def sqrt_(a: Tensor, /) -> Tensor:       return _inplace1(_C_engine.sqrt_, a)
def sin_(a: Tensor, /) -> Tensor:        return _inplace1(_C_engine.sin_, a)
def cos_(a: Tensor, /) -> Tensor:        return _inplace1(_C_engine.cos_, a)
def tan_(a: Tensor, /) -> Tensor:        return _inplace1(_C_engine.tan_, a)
def arcsin_(a: Tensor, /) -> Tensor:     return _inplace1(_C_engine.arcsin_, a)
def arccos_(a: Tensor, /) -> Tensor:     return _inplace1(_C_engine.arccos_, a)
def arctan_(a: Tensor, /) -> Tensor:     return _inplace1(_C_engine.arctan_, a)
def sinh_(a: Tensor, /) -> Tensor:       return _inplace1(_C_engine.sinh_, a)
def cosh_(a: Tensor, /) -> Tensor:       return _inplace1(_C_engine.cosh_, a)
def tanh_(a: Tensor, /) -> Tensor:       return _inplace1(_C_engine.tanh_, a)
def round_(a: Tensor, /, decimals: int = 0) -> Tensor:
    if decimals != 0:
        a._impl = round(a, decimals)._impl
        return a
    return _inplace1(_C_engine.round_, a)
def floor_(a: Tensor) -> Tensor:         return _inplace1(_C_engine.floor_, a)
def ceil_(a: Tensor) -> Tensor:          return _inplace1(_C_engine.ceil_, a)


def clip_(a: Tensor, /, min_value=None, max_value=None) -> Tensor:
    if min_value is None: min_value = min(a).item()
    if max_value is None: max_value = max(a).item()
    _C_engine.clip_(impl_of(a), float(min_value), float(max_value))
    return a
