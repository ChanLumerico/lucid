from typing import Optional

import lucid
from lucid._tensor import Tensor
from lucid.nn import Parameter


# linear functions
def linear(
    input: Tensor, weight: Parameter, bias: Optional[Parameter] = None
) -> Tensor:
    output = input @ weight.T
    if bias is not None:
        output += bias

    return output


# non-linear activation functions
def relu(input: Tensor) -> Tensor:
    return input * (input > 0)


def sigmoid(input: Tensor) -> Tensor:
    return 1 / (1 + lucid.exp(-input))


def tanh(input: Tensor) -> Tensor:
    return lucid.tanh(input)


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    exp_input = lucid.exp(input)
    return exp_input / exp_input.sum(axis=dim, keepdims=True)
