from functools import reduce

import lucid

from lucid._tensor import Tensor
from lucid.types import _Scalar


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    if tensor.ndim == 2:
        fan_in = tensor.shape[0]
        fan_out = tensor.shape[1]

    elif tensor.ndim in {3, 4, 5}:
        kernel_prod = reduce(lambda x, y: x * y, tensor.shape[2:], 1)
        fan_in = tensor.shape[1] * kernel_prod
        fan_out = tensor.shape[0] * kernel_prod

    else:
        raise ValueError(
            f"Tensor with dims {tensor.ndim} is not supported. "
            + "Must be at least 2D."
        )

    return fan_in, fan_out


def uniform_(tensor: Tensor, a: _Scalar, b: _Scalar) -> None:
    tensor.data = lucid.random.uniform(a, b, tensor.shape).data


def normal_(tensor: Tensor, mean: _Scalar, std: _Scalar) -> None:
    tensor.data = lucid.random.randn(tensor.shape).data * std + mean


def constant_(tensor: Tensor, val: _Scalar) -> None:
    tensor.data = lucid.ones_like(tensor).data * val


def xavier_uniform_(tensor: Tensor, gain: _Scalar) -> None: ...  # TODO: Begin from here
