"""
lucid
=====
Lumeruco's Comprehensive Interface for Deep Learning
"""

from contextlib import contextmanager
from typing import Any, Generator
import numpy as np

from lucid._tensor import Tensor
from lucid._func import *
from lucid._util import *

from lucid.types import _ArrayOrScalar

import lucid.linalg as linalg
import lucid.random as random
import lucid.nn as nn

_grad_enabled: bool = True

newaxis = np.newaxis


def tensor(
    data: Tensor | _ArrayOrScalar, requires_grad: bool = False, dtype: Any = np.float32
) -> Tensor:
    if isinstance(data, Tensor):
        data = data.data
    return Tensor(data, requires_grad, dtype)


@contextmanager
def no_grad() -> Generator:
    global _grad_enabled
    prev_state = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = prev_state


def grad_enabled() -> bool:
    return _grad_enabled
