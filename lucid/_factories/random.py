"""
Random tensor creation: rand, randn, randint, bernoulli, normal, manual_seed.
"""

import os
from typing import TYPE_CHECKING

import lucid as _lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import normalize_factory_kwargs, _wrap, _impl_with_grad
from lucid._dtype import dtype, int64
from lucid._types import DeviceLike, DTypeLike

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


# Re-export the engine class so users can write ``lucid.Generator(seed)``
# without reaching into ``lucid._C``.
Generator = _C_engine.Generator


_default_generator: _C_engine.Generator | None = None


def _active_default_gen() -> _C_engine.Generator:
    """Return the generator that random ops *actually* read from.  Lazy-
    initialised on first access to mirror the C++ singleton's seed=0 default."""
    global _default_generator
    if _default_generator is None:
        _default_generator = _C_engine.Generator(0)
    return _default_generator


def manual_seed(seed: int) -> None:
    """Set the seed for the default random number generator."""
    global _default_generator
    _default_generator = _C_engine.Generator(seed)
    # Keep the C++ singleton in sync — used by engine code paths that don't
    # take an explicit ``generator`` arg.
    _C_engine.default_generator().set_seed(seed)


def seed() -> int:
    """Seed the default generator from the OS entropy source and return
    the seed used.  Mirrors the reference framework's contract: subsequent
    sampling is non-deterministic across processes."""
    s: int = int.from_bytes(os.urandom(8), "little", signed=False)
    # Mask to the 63-bit non-negative range so the int round-trips through
    # signed APIs cleanly.
    s &= 0x7FFF_FFFF_FFFF_FFFF
    manual_seed(s)
    return s


def initial_seed() -> int:
    """Return the seed of the default generator (whatever it was last set
    to via :func:`manual_seed` / :func:`seed`)."""
    return int(_active_default_gen().seed)


def get_rng_state() -> Tensor:
    """Return the current default generator's state as an ``int64`` tensor
    of length 2: ``[seed, counter]``.

    This is the minimal serialisation that round-trips through
    :func:`set_rng_state` — the Philox PRNG is fully determined by the
    pair, so a longer "state vector" (which the reference framework uses
    for its mt19937 backend) is unnecessary here.
    """
    g = _active_default_gen()
    return _lucid.tensor([int(g.seed), int(g.counter)], dtype=int64)


def set_rng_state(state: Tensor) -> None:
    """Restore the default generator's state from a tensor previously
    produced by :func:`get_rng_state`.  Must be a length-2 ``int64``
    tensor ``[seed, counter]``."""
    global _default_generator
    if int(state.numel()) != 2:
        raise ValueError(
            f"set_rng_state: expected a length-2 state tensor, "
            f"got shape={tuple(state.shape)}"
        )
    flat = state.reshape(-1)
    s_seed = int(flat[0].item())
    s_counter = int(flat[1].item())
    # Build a fresh generator at the captured (seed, counter) pair — Philox is
    # counter-based so this exactly recovers the prior sampling stream.
    g = _C_engine.Generator(s_seed)
    g.counter = s_counter
    _default_generator = g
    # Mirror to the C++ singleton.
    cg = _C_engine.default_generator()
    cg.set_seed(s_seed)
    cg.counter = s_counter


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
    return _wrap(
        _impl_with_grad(perm_impl, requires_grad) if requires_grad else perm_impl
    )
