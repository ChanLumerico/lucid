from typing import Optional

import lucid
from lucid._tensor import Tensor
from lucid.nn import Parameter


def linear(
    input: Tensor, weight: Parameter, bias: Optional[Parameter] = None
) -> Tensor:
    output = input @ weight.T
    if bias is not None:
        output += bias

    return output


def relu(input: Tensor) -> Tensor:
    return input * (input > 0)


def sigmoid(input: Tensor) -> Tensor:
    return 1 / (1 + lucid.exp(-input))
