from lucid._func import ufunc
from lucid._tensor.tensor import Tensor
from lucid._backend.core import _GradType

from .base import FusedBackwardOp


class NegNeg(FusedBackwardOp):
    op1 = ufunc._neg
    op2 = ufunc._neg

    @classmethod
    def __grad__(cls, rets: tuple[Tensor]) -> _GradType:
        ret = rets[0]
        return ret.grad
