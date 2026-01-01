from lucid._tensor.tensor import Tensor
from lucid._backend.core import _GradType

from lucid._func import ufunc
from lucid._util import func as util_func

from .base import FusedBackwardOp


__all__ = [
    "DoubleNeg",
    "DoubleReciprocal",
    "LogExp",
    "DoubleT",
    "DoubleMT",
    "DoubleReshape",
    "SqueezeUnsqueeze",
    "UnsqueezeSqueeze",
]


class _IdentityFusion(FusedBackwardOp):
    @classmethod
    def __grad__(cls, rets: tuple[Tensor]) -> _GradType:
        return rets[0].grad


class _IdentityViewFusion(FusedBackwardOp):
    @classmethod
    def __grad__(cls, ins: tuple[Tensor], rets: tuple[Tensor]) -> _GradType:
        v = rets[0]
        x = ins[0]
        return v.grad.reshape(x.shape) if v.grad is not None else None


class DoubleNeg(_IdentityFusion):
    op1 = ufunc._neg
    op2 = ufunc._neg


class DoubleReciprocal(_IdentityFusion):
    op1 = ufunc.reciprocal
    op2 = ufunc.reciprocal


class LogExp(_IdentityFusion):
    op1 = ufunc.exp
    op2 = ufunc.log

    heuristic_thresh = 10_000


class DoubleT(_IdentityFusion):
    op1 = ufunc._T
    op2 = ufunc._T


class DoubleMT(_IdentityFusion):
    op1 = ufunc._mT
    op2 = ufunc._mT


class DoubleReshape(_IdentityViewFusion):
    op1 = util_func.reshape
    op2 = util_func.reshape


class _DoubleReshapeImmediate(_IdentityViewFusion):
    op1 = getattr(util_func, "_reshape_immediate", None)
    op2 = getattr(util_func, "_reshape_immediate", None)


class SqueezeUnsqueeze(_IdentityViewFusion):
    op1 = util_func.squeeze
    op2 = util_func.unsqueeze


class UnsqueezeSqueeze(_IdentityViewFusion):
    op1 = util_func.unsqueeze
    op2 = util_func.squeeze
