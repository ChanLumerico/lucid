"""
Random tensor creation: rand, randn, randint, bernoulli, normal, manual_seed.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import normalize_factory_kwargs, _wrap, _impl_with_grad
from lucid._dtype import dtype, int64
from lucid._types import DeviceLike, DTypeLike

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


_default_generator: _C_engine.Generator | None = None


def manual_seed(seed: int) -> None:
    """Set the seed for the default random number generator."""
    global _default_generator
    _default_generator = _C_engine.Generator(seed)
    _C_engine.default_generator().set_seed(seed)


def _get_gen(
    generator: _C_engine.Generator | None,
) -> _C_engine.Generator | None:
    return generator if generator is not None else _default_generator


def rand(
    *size: int | tuple[int, ...],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    """Return a tensor filled with uniform random values in [0, 1)."""
    _dt, _dev, _ = normalize_factory_kwargs(dtype, device)
    shape = _size_to_list(*size)
    impl = _C_engine.rand(shape, _dt, _dev, _get_gen(generator))
    return _wrap(_impl_with_grad(impl, requires_grad) if requires_grad else impl)


def randn(
    *size: int | tuple[int, ...],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    """Return a tensor filled with standard normal random values."""
    _dt, _dev, _ = normalize_factory_kwargs(dtype, device)
    shape = _size_to_list(*size)
    impl = _C_engine.randn(shape, _dt, _dev, _get_gen(generator))
    return _wrap(_impl_with_grad(impl, requires_grad) if requires_grad else impl)


def randint(
    low: int,
    high: int,
    size: list[int] | tuple[int, ...],
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    """Return a tensor filled with random integers in [low, high)."""
    _dt, _dev, _ = normalize_factory_kwargs(
        dtype if dtype is not None else int64, device
    )
    shape = list(size)
    impl = _C_engine.randint(shape, low, high, _dt, _dev, _get_gen(generator))
    return _wrap(_impl_with_grad(impl, requires_grad) if requires_grad else impl)


def bernoulli(
    p: float,
    *,
    size: list[int] | tuple[int, ...] | None = None,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    """Return a tensor of Bernoulli samples with probability p."""
    _dt, _dev, _ = normalize_factory_kwargs(dtype, device)
    shape = list(size) if size is not None else [1]
    impl = _C_engine.bernoulli(shape, p, _dt, _dev, _get_gen(generator))
    return _wrap(_impl_with_grad(impl, requires_grad) if requires_grad else impl)


def normal(
    mean: float = 0.0,
    std: float = 1.0,
    *,
    size: list[int] | tuple[int, ...],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    """Return a tensor filled with normal random values."""
    _dt, _dev, _ = normalize_factory_kwargs(dtype, device)
    shape = list(size)
    impl = _C_engine.normal(shape, mean, std, _dt, _dev, _get_gen(generator))
    return _wrap(_impl_with_grad(impl, requires_grad) if requires_grad else impl)


def rand_like(
    t: Tensor,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    """Return a uniform random tensor with the same shape/dtype/device as t."""
    _dt, _dev, _ = normalize_factory_kwargs(
        dtype if dtype is not None else t.dtype,
        device if device is not None else t.device,
    )
    impl = _C_engine.rand(list(t.shape), _dt, _dev, None)
    return _wrap(_impl_with_grad(impl, requires_grad) if requires_grad else impl)


def randn_like(
    t: Tensor,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    """Return a normal random tensor with the same shape/dtype/device as t."""
    _dt, _dev, _ = normalize_factory_kwargs(
        dtype if dtype is not None else t.dtype,
        device if device is not None else t.device,
    )
    impl = _C_engine.randn(list(t.shape), _dt, _dev, None)
    return _wrap(_impl_with_grad(impl, requires_grad) if requires_grad else impl)


from lucid._factories.creation import _size_to_list


def randperm(
    n: int,
    *,
    generator: _C_engine.Generator | None = None,
    dtype: DTypeLike = int64,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    """Return a random permutation of integers in ``[0, n)``.

    Implemented as ``argsort(rand(n))`` — the lottery-ticket trick: each
    element gets a uniform random key, and the argsort of those keys is
    a uniformly random permutation.  Output is int64 by default to match
    the standard reference framework.
    """
    if n < 0:
        raise ValueError(f"randperm requires n >= 0, got {n}")
    _dt, _dev, _ = normalize_factory_kwargs(dtype, device)
    if n == 0:
        impl = _C_engine.zeros([0], _dt, _dev)
        return _wrap(_impl_with_grad(impl, requires_grad) if requires_grad else impl)
    keys_impl = _C_engine.rand([n], _C_engine.F32, _dev, _get_gen(generator))
    perm_impl = _C_engine.argsort(keys_impl, -1)
    if perm_impl.dtype != _dt:
        perm_impl = _C_engine.astype(perm_impl, _dt)
    return _wrap(_impl_with_grad(perm_impl, requires_grad) if requires_grad else perm_impl)
