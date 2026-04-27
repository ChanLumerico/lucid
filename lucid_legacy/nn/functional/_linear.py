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
    outputs = []
    for i in range(weight.shape[0]):
        wi = weight[i]
        outputs.append(((input_1 @ wi) * input_2).sum(axis=1, keepdims=True))

    if len(outputs) == 1:
        output = outputs[0]
    else:
        output = lucid.concatenate(tuple(outputs), axis=1)

    if bias is not None:
        output += bias

    return output
