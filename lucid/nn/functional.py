from typing import Optional

from lucid._tensor import Tensor
from lucid.nn import Parameter


def linear(
    input: Tensor, weight: Parameter, bias: Optional[Parameter] = None
) -> Tensor:
    output = input @ weight.T
    if bias is not None:
        output += bias

    return output
