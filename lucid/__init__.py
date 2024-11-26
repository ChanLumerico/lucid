"""
# `Lucid `

**Lucid** is an educational deep learning framework developed to help users understand 
the underlying mechanics of deep learning models and tensor operations. 

It is designed to provide a simple yet powerful environment to experiment with neural networks, 
optimization, and backpropagation using only `NumPy`. 

Lucid is ideal for those who want to learn about the inner workings of deep learning 
algorithms and operations without the complexity of high-level frameworks.

[📑 Lucid Documentation](https://chanlumerico.github.io/lucid/build/html/index.html)
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

pi = np.pi
inf = np.inf


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
