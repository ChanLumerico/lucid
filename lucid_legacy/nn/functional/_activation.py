import lucid

from lucid._tensor import Tensor
from lucid.nn._kernel.activation import (
    softmax_kernel,
    sigmoid_kernel,
    gelu_kernel,
    silu_kernel,
)


def relu(input_: Tensor) -> Tensor:
    return lucid.maximum(0, input_)


def leaky_relu(input_: Tensor, negative_slope: float = 0.01) -> Tensor:
    mask = input_ > 0
    out = input_ * mask + input_ * negative_slope * (~mask)
    return out


def elu(input_: Tensor, alpha: float = 1.0) -> Tensor:
    mask = input_ >= 0
    pos = input_ * mask
    neg = alpha * (lucid.exp(input_) - 1) * (~mask)
    return pos + neg


def selu(input_: Tensor) -> Tensor:
    _scale = 1.0507009873554805
    _alpha = 1.6732632423543772

    mask = input_ >= 0
    pos = _scale * input_ * mask
    neg = _scale * _alpha * (lucid.exp(input_) - 1) * (~mask)
    return pos + neg


def gelu(input_: Tensor) -> Tensor:
    op = gelu_kernel()
    return op(input_)


def sigmoid(input_: Tensor) -> Tensor:
    op = sigmoid_kernel()
    return op(input_)


def tanh(input_: Tensor) -> Tensor:
    return lucid.tanh(input_)


def silu(input_: Tensor) -> Tensor:
    op = silu_kernel()
    return op(input_)


def softmax(input_: Tensor, axis: int = -1) -> Tensor:
    op = softmax_kernel(axis=axis)
    return op(input_)
