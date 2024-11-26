import lucid

from lucid._tensor import Tensor


def relu(input_: Tensor) -> Tensor:
    return lucid.maximum(0, input_)


# TODO: resolve for (1 - mask), the `int` 1 not being treated as a `Tensor`.
def leaky_relu(input_: Tensor, negative_slope: float = 0.01) -> Tensor:
    mask = input_ > 0
    out = input_ * mask + input_ * negative_slope * (1 - mask)
    return out


def elu(input_: Tensor, alpha: float = 1.0) -> Tensor:
    mask = input_ >= 0
    pos = input_ * mask
    neg = alpha * (lucid.exp(input_) - 1) * (1 - mask)
    return pos + neg


def selu(input_: Tensor) -> Tensor:
    _scale = 1.0507009873554805
    _alpha = 1.6732632423543772

    mask = input_ >= 0
    pos = _scale * input_ * mask
    neg = _scale * _alpha * (lucid.exp(input_) - 1) * (1 - mask)
    return pos + neg


def sigmoid(input_: Tensor) -> Tensor:
    return 1 / (1 + lucid.exp(input_))


def tanh(input_: Tensor) -> Tensor:
    return lucid.tanh(input_)
