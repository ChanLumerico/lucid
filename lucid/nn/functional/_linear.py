"""
lucid.nn.functional._linear — linear and bilinear.

`linear` routes to the engine's fused linear (`x @ W^T + b`).
`bilinear` is Python composition (small surface, no engine kernel).
"""

from __future__ import annotations

import lucid

from lucid._C.engine import nn as _C_nn
from lucid._tensor import Tensor
from lucid._bridge import impl_of


def linear(input_: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    if bias is None:
        # Engine requires bias; emulate with x @ W^T.
        return input_ @ weight.mT
    return Tensor._wrap(_C_nn.linear(impl_of(input_), impl_of(weight), impl_of(bias)))


def bilinear(
    input_1: Tensor, input_2: Tensor, weight: Tensor, bias: Tensor | None = None
) -> Tensor:
    outputs = []
    for i in range(weight.shape[0]):
        wi = weight[i]
        outputs.append(((input_1 @ wi) * input_2).sum(axis=1, keepdims=True))

    if len(outputs) == 1:
        output = outputs[0]
    else:
        output = lucid.concatenate(tuple(outputs), axis=1)

    if bias is not None:
        output = output + bias
    return output
