"""
lucid
=====
Lumeruco's Comprehensive Interface for Deep Learning
"""

import lucid

from lucid.tensor import Tensor, _ArrayOrScalar
from lucid.op import *


_basic_ops = (
    "add",
    "radd",
    "sub",
    "rsub",
    "mul",
    "rmul",
    "truediv",
    "rtruediv",
    "pow",
)

for _op in _basic_ops:
    setattr(Tensor, f"__{_op}__", getattr(lucid._func, _op))


def tensor(data: _ArrayOrScalar, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad)
