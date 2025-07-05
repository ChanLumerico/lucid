import numpy as np

import lucid
from lucid._tensor import Tensor
from lucid.types import (
    _ShapeLike,
    _Scalar,
    _ArrayOrScalar,
    _DeviceType,
    _BuiltinNumeric,
    Numeric,
)

from lucid._backend.metal import mx


def seed(seed: int) -> None:
    np.random.seed(seed)
    mx.random.seed(seed)


def rand(
    shape: _ShapeLike,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return Tensor(np.random.rand(*shape), requires_grad, keep_grad, device=device)


def randint(
    low: int,
    high: int | None,
    size: int | _ShapeLike,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return Tensor(
        np.random.randint(low, high, size),
        requires_grad,
        keep_grad,
        lucid.Int64,
        device,
    )


def randn(
    shape: _ShapeLike,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return Tensor(np.random.randn(*shape), requires_grad, keep_grad, device=device)


def uniform(
    low: _Scalar,
    high: _Scalar,
    size: int | _ShapeLike,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return Tensor(
        np.random.uniform(low, high, size), requires_grad, keep_grad, device=device
    )


def bernoulli(
    probs: _ArrayOrScalar | Tensor,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    if isinstance(probs, Tensor):
        probs_data = probs.data
    else:
        probs_data = np.array(probs)

    if np.any(probs_data < 0) or np.any(probs_data > 1):
        raise ValueError("probs must be in the range [0, 1].")

    return Tensor(
        (np.random.rand(*probs_data.shape) < probs_data).astype(int),
        requires_grad,
        keep_grad,
        device=device,
    )


def permutation(
    n: int,
    dtype: _BuiltinNumeric | Numeric,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    data = np.random.permutation(n)
    return Tensor(data, requires_grad, keep_grad, dtype, device)
