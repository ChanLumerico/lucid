import lucid

from lucid._tensor import Tensor


def linear(input_: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    output = input_ @ weight.mT
    if bias is not None:
        output += bias

    return output


def bilinear(
    input_1: Tensor, input_2: Tensor, weight: Tensor, bias: Tensor | None = None
) -> Tensor:
    x1_outer_x2 = input_1[..., :, lucid.newaxis] * input_2[..., lucid.newaxis, :]
    x1_outer_x2_flat = x1_outer_x2.reshape(*x1_outer_x2.shape[:-2], -1)

    weight_flat = weight.reshape(weight.shape[0], -1)
    output = x1_outer_x2_flat @ weight_flat.T

    if bias is not None:
        output += bias

    return output
