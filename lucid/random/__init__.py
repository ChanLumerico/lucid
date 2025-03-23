from typing import overload

import lucid
from lucid.random import _func
from lucid._tensor import Tensor
from lucid.types import _ShapeLike, _Scalar, _ArrayOrScalar, _DeviceType


# fmt: off
__all__ = [
    "seed", "rand", "randint", "randn", "uniform", "bernoulli"
]
# fmt: on


def seed(seed: int) -> None:
    return _func.seed(seed)


@overload
def rand(
    *shape: int,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu"
) -> Tensor: ...


@overload
def rand(
    shape: _ShapeLike,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor: ...


def rand(
    *args: int,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu"
) -> Tensor:
    shape = lucid._get_overloaded_shape(args)
    return _func.rand(shape, requires_grad, keep_grad, device)


def randint(
    low: int,
    high: int | None,
    size: int | _ShapeLike = 1,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return _func.randint(low, high, size, requires_grad, keep_grad, device)


@overload
def randn(
    *shape: int,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu"
) -> Tensor: ...


@overload
def randn(
    shape: _ShapeLike,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor: ...


def randn(
    *args: int | _ShapeLike,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu"
) -> Tensor:
    shape = lucid._get_overloaded_shape(args)
    return _func.randn(shape, requires_grad, keep_grad, device)


def uniform(
    low: _Scalar = 0,
    high: _Scalar = 1,
    size: int | _ShapeLike = 1,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return _func.uniform(low, high, size, requires_grad, keep_grad, device)


def bernoulli(
    probs: _ArrayOrScalar | Tensor,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return _func.bernoulli(probs, requires_grad, keep_grad, device)
