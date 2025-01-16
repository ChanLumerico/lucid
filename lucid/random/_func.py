import numpy as np

from lucid._tensor import Tensor
from lucid.types import _ShapeLike, _Scalar, _ArrayOrScalar


def seed(seed: int) -> None:
    np.random.seed(seed)


def rand(
    shape: _ShapeLike, requires_grad: bool = False, keep_grad: bool = False
) -> Tensor:
    return Tensor(np.random.rand(*shape), requires_grad, keep_grad)


def randint(
    low: int,
    high: int | None,
    size: int | _ShapeLike,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    return Tensor(np.random.randint(low, high, size), requires_grad, keep_grad)


def randn(
    shape: _ShapeLike, requires_grad: bool = False, keep_grad: bool = False
) -> Tensor:
    return Tensor(np.random.randn(*shape), requires_grad, keep_grad)


def uniform(
    low: _Scalar,
    high: _Scalar,
    size: int | _ShapeLike,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    return Tensor(np.random.uniform(low, high, size), requires_grad, keep_grad)


def bernoulli(
    probs: _ArrayOrScalar | Tensor,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    if isinstance(probs, Tensor):
        probs_data = probs.data
    else:
        probs_data = np.array(probs)

    if np.any(probs_data < 0) or np.any(probs_data > 1):
        raise ValueError("probs must be in the range [0, 1].")

    return Tensor(
        np.random.rand(*probs_data.shape) < probs_data, requires_grad, keep_grad
    )
