"""
lucid
=====
Lumeruco's Comprehensive Interface for Deep Learning
"""

from typing import Any
import numpy as np

from lucid.tensor import Tensor, _ArrayOrScalar
from lucid._func import *


def tensor(
    data: _ArrayOrScalar, requires_grad: bool = False, dtype: Any = np.float32
) -> Tensor:
    return Tensor(data, requires_grad, dtype)
