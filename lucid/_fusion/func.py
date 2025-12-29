from lucid._func import ufunc
from lucid._tensor.tensor import Tensor
from lucid._backend.core import _GradType

from .base import FusedBackwardOp


__all__ = ["DoubleNeg", "DoubleReciprocal", "LogExp", "DoubleT", "DoubleMT"]


class _IdentityFusion(FusedBackwardOp):
    @classmethod
    def __grad__(cls, rets: tuple[Tensor]) -> _GradType:
        return rets[0].grad


class DoubleNeg(_IdentityFusion):
    op1 = ufunc._neg
    op2 = ufunc._neg


class DoubleReciprocal(_IdentityFusion):
    op1 = ufunc.reciprocal
    op2 = ufunc.reciprocal


class LogExp(_IdentityFusion):
    op1 = ufunc.exp
    op2 = ufunc.log


class DoubleT(_IdentityFusion):
    op1 = ufunc._T
    op2 = ufunc._T


class DoubleMT(_IdentityFusion):
    op1 = ufunc._mT
    op2 = ufunc._mT
