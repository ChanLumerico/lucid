from lucid._tensor import Tensor

from lucid.nn.functional import _linear, _non_linear


def linear(input_: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    return _linear.linear(input_, weight, bias)


def bilinear(
    input_1: Tensor, input_2: Tensor, weight: Tensor, bias: Tensor | None = None
) -> Tensor:
    return _linear.bilinear(input_1, input_2, weight, bias)


def relu(input_: Tensor) -> Tensor:
    return _non_linear.relu(input_)


def leaky_relu(input_: Tensor, negative_slope: float = 0.01) -> Tensor:
    return _non_linear.leaky_relu(input_, negative_slope)


def elu(input_: Tensor, alpha: float = 1.0) -> Tensor:
    return _non_linear.elu(input_, alpha)


def selu(input_: Tensor) -> Tensor:
    return _non_linear.selu(input_)


def gelu(input_: Tensor) -> Tensor:
    return _non_linear.gelu(input_)


def sigmoid(input_: Tensor) -> Tensor:
    return _non_linear.sigmoid(input_)


def tanh(input_: Tensor) -> Tensor:
    return _non_linear.tanh(input_)
