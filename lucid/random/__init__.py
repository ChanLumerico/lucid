from lucid.random import func

from lucid._tensor import Tensor
from lucid.types import _ShapeLike


def seed(seed: int) -> None:
    return func.seed(seed)


def rand(*shape: int, requires_grad: bool = False, keep_grad: bool = False) -> Tensor:
    return func.rand(*shape, requires_grad=requires_grad, keep_grad=keep_grad)


def randint(
    low: int,
    high: int | None,
    size: int | _ShapeLike,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    return func.randint(low, high, size, requires_grad, keep_grad)


def randn(*shape: int, requires_grad: bool = False, keep_grad: bool = False) -> Tensor:
    return func.randn(*shape, requires_grad=requires_grad, keep_grad=keep_grad)
