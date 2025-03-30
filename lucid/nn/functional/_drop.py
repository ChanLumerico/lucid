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

    mask = (lucid.random.rand(*input_.shape) > p).free()
    scale = 1.0 / (1 - p)
    return input_ * mask * scale


def dropoutnd(input_: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    _prob_check(p)
    if not training:
        return input_

    spatial_dim = input_.ndim - 2
    mask = (lucid.random.rand(*input_.shape[:2], *(1,) * spatial_dim) > p).free()
    scale = 1.0 / (1 - p)
    return input_ * mask * scale


def alpha_dropout(input_: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    _prob_check(p)
    if not training:
        return input_

    _alpha = -1.7580993408473766
    _lambda = 1.0507009873554805

    mask = (lucid.random.rand(*input_.shape) > p).free()
    scale = 1.0 / (1 - p)

    dropped = input_ * mask * scale
    noise = (1 - mask) * _alpha * _lambda
    return dropped + noise


def drop_block(
    input_: Tensor, block_size: int, p: float = 0.1, eps: float = 1e-7
) -> Tensor:
    _, _, h, w = input_.shape
    gamma = (
        p
        * (h * w)
        / (block_size**2)
        / ((h - block_size + 1) * (w - block_size + 1) + eps)
    )
    mask = (lucid.random.rand(*input_.shape) < gamma).astype(float)

    pad = block_size // 2
    padded_mask = lucid.pad(mask, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    block_mask = lucid.zeros_like(mask)

    for i in range(block_size):
        for j in range(block_size):
            block_mask = lucid.maximum(
                block_mask, padded_mask[:, :, i : i + h, j : j + w]
            )

    block_mask = (1 - block_mask).free()
    return input_ * block_mask / (1 - p)


def drop_path(input_: Tensor, p: float = 0.0, scale_by_keep: bool = True) -> Tensor:
    if p == 0.0:
        return input_

    keep_prob = 1 - p
    shape = (input_.shape[0],) + (1,) * (input_.ndim - 1)

    random_tensor = lucid.random.bernoulli(keep_prob * lucid.ones(shape)).free()
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor /= keep_prob

    return input_ * random_tensor
