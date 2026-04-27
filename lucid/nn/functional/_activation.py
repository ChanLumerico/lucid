"""
lucid.nn.functional._activation — activation functions.

All routes are 1:1 to C++ engine ops — no Python compositions.
"""

from __future__ import annotations

from lucid._C import engine as _C_engine
from lucid._tensor import Tensor
from lucid._bridge import impl_of


def relu(input_: Tensor) -> Tensor:
    return Tensor._wrap(_C_engine.relu(impl_of(input_)))


def leaky_relu(input_: Tensor, negative_slope: float = 0.01) -> Tensor:
    return Tensor._wrap(_C_engine.leaky_relu(impl_of(input_), float(negative_slope)))


def elu(input_: Tensor, alpha: float = 1.0) -> Tensor:
    return Tensor._wrap(_C_engine.elu(impl_of(input_), float(alpha)))


def selu(input_: Tensor) -> Tensor:
    return Tensor._wrap(_C_engine.selu(impl_of(input_)))


def gelu(input_: Tensor) -> Tensor:
    return Tensor._wrap(_C_engine.gelu(impl_of(input_)))


def sigmoid(input_: Tensor) -> Tensor:
    return Tensor._wrap(_C_engine.sigmoid(impl_of(input_)))


def tanh(input_: Tensor) -> Tensor:
    return Tensor._wrap(_C_engine.tanh(impl_of(input_)))


def silu(input_: Tensor) -> Tensor:
    return Tensor._wrap(_C_engine.silu(impl_of(input_)))


def softmax(input_: Tensor, axis: int = -1) -> Tensor:
    return Tensor._wrap(_C_engine.softmax(impl_of(input_), int(axis)))


def softplus(input_: Tensor) -> Tensor:
    return Tensor._wrap(_C_engine.softplus(impl_of(input_)))


def mish(input_: Tensor) -> Tensor:
    return Tensor._wrap(_C_engine.mish(impl_of(input_)))


def hard_sigmoid(input_: Tensor) -> Tensor:
    return Tensor._wrap(_C_engine.hard_sigmoid(impl_of(input_)))


def hard_swish(input_: Tensor) -> Tensor:
    return Tensor._wrap(_C_engine.hard_swish(impl_of(input_)))


def relu6(input_: Tensor) -> Tensor:
    return Tensor._wrap(_C_engine.relu6(impl_of(input_)))
