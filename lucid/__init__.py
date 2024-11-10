"""
lucid
=====
Lumeruco's Comprehensive Interface for Deep Learning
"""

from typing import Any
import numpy as np

from lucid._tensor import Tensor
from lucid._func import *

from lucid.types import _ArrayOrScalar

import lucid.linalg as linalg
import lucid.random as random
import lucid.nn as nn


def tensor(
    data: Tensor | _ArrayOrScalar, requires_grad: bool = False, dtype: Any = np.float32
) -> Tensor:
    if isinstance(data, Tensor):
        data = data.data
    return Tensor(data, requires_grad, dtype)
