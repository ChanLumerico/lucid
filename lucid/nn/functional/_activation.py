import lucid

from lucid._tensor import Tensor


def relu(input_: Tensor) -> Tensor:
    return lucid.maximum(0, input_)


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


def gelu(input_: Tensor) -> Tensor:
    c = lucid.sqrt(2 / lucid.pi)
    return 0.5 * input_ * (1 + lucid.tanh(c * (input_ + 0.044715 * input_**3)))


def sigmoid(input_: Tensor) -> Tensor:
    return 1 / (1 + lucid.exp(input_))


def tanh(input_: Tensor) -> Tensor:
    return lucid.tanh(input_)


def softmax(input_: Tensor, axis: int = -1) -> Tensor:
    input_max = lucid.max(input_, axis=axis, keepdims=True)
    input_stable = input_ - input_max

    e_input = lucid.exp(input_stable)
    sum_e_input = e_input.sum(axis=axis, keepdims=True)

    output = e_input / sum_e_input
    return output
