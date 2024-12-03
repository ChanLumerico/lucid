from typing import Never
import lucid
from lucid._tensor import Tensor


def _prob_check(p: float) -> Never:
    if not 0 <= p < 1:
        raise ValueError("Dropout probability `p` must be in the range [0, 1).")


def dropout(input_: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    _prob_check(p)
    if not training:
        return input_

    mask = lucid.random.rand(*input_.shape) > p
    scale = 1.0 / (1 - p)
    return input_ * mask * scale


def dropoutnd(input_: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    _prob_check(p)
    if not training:
        return input_

    spatial_dim = input_.ndim - 2
    mask = lucid.random.rand(*input_.shape[:2], *(1,) * spatial_dim) > p
    scale = 1.0 / (1 - p)
    return input_ * mask * scale


def alpha_dropout(input_: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    _prob_check(p)
    if not training:
        return input_

    _alpha = -1.7580993408473766
    _lambda = 1.0507009873554805

    mask = lucid.random.rand(*input_.shape) > p
    scale = 1.0 / (1 - p)

    dropped = input_ * mask * scale
    noise = (1 - mask) * _alpha * _lambda
    return dropped + noise
