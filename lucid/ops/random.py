"""
lucid.ops.random — random tensor sampling (mirrors `lucid_legacy/random/`).

All draws route through Lucid's Philox-4x32-10 generator, so the same
seed yields bit-exact identical output across CPU and GPU.
"""

from __future__ import annotations

from lucid._C import engine as _C_engine
from lucid._tensor import Tensor
from lucid._bridge import (
    normalize_shape, to_engine_dtype, to_engine_device,
)
from lucid.types import (
    Numeric, _BuiltinNumeric, _DeviceType, _ShapeLike,
)


__all__ = [
    "Generator", "default_generator",
    "manual_seed", "seed", "set_deterministic", "is_deterministic",
    "rand", "randn", "uniform", "normal", "randint", "bernoulli",
    "permutation",
]


class Generator:
    """User-facing wrapper around the C++ Philox `_C_engine.Generator`."""

    def __init__(self, seed: int = 0) -> None:
        self._cpp = _C_engine.Generator(int(seed))

    def set_seed(self, seed: int) -> None:
        self._cpp.set_seed(int(seed))

    @property
    def seed(self) -> int:
        return self._cpp.seed

    @property
    def counter(self) -> int:
        return self._cpp.counter

    def __repr__(self) -> str:
        return f"Generator(seed={self.seed}, counter={self.counter})"


def default_generator() -> "_C_engine.Generator":
    return _C_engine.default_generator()


def manual_seed(seed: int) -> None:
    """Seed the process-default generator (matches PyTorch convention)."""
    default_generator().set_seed(int(seed))


def set_deterministic(value: bool) -> None:
    _C_engine.set_deterministic(bool(value))


def is_deterministic() -> bool:
    return _C_engine.is_deterministic()


def _gen_obj(g: Generator | None):
    return g._cpp if isinstance(g, Generator) else None


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

def rand(
    *shape: int | _ShapeLike,
    dtype: _BuiltinNumeric | Numeric | None = None,
    device: _DeviceType = "cpu",
    generator: Generator | None = None,
    requires_grad: bool = False,
) -> Tensor:
    sh = normalize_shape(shape)
    impl = _C_engine.rand(sh, to_engine_dtype(dtype),
                          to_engine_device(device), _gen_obj(generator))
    t = Tensor._wrap(impl)
    if requires_grad:
        t.requires_grad_(True)
    return t


def randn(
    *shape: int | _ShapeLike,
    dtype: _BuiltinNumeric | Numeric | None = None,
    device: _DeviceType = "cpu",
    generator: Generator | None = None,
    requires_grad: bool = False,
) -> Tensor:
    sh = normalize_shape(shape)
    impl = _C_engine.randn(sh, to_engine_dtype(dtype),
                           to_engine_device(device), _gen_obj(generator))
    t = Tensor._wrap(impl)
    if requires_grad:
        t.requires_grad_(True)
    return t


def uniform(
    *shape: int | _ShapeLike,
    low: float = 0.0, high: float = 1.0,
    dtype: _BuiltinNumeric | Numeric | None = None,
    device: _DeviceType = "cpu",
    generator: Generator | None = None,
    requires_grad: bool = False,
) -> Tensor:
    sh = normalize_shape(shape)
    impl = _C_engine.uniform(sh, float(low), float(high),
                              to_engine_dtype(dtype),
                              to_engine_device(device), _gen_obj(generator))
    t = Tensor._wrap(impl)
    if requires_grad:
        t.requires_grad_(True)
    return t


def normal(
    *shape: int | _ShapeLike,
    mean: float = 0.0, std: float = 1.0,
    dtype: _BuiltinNumeric | Numeric | None = None,
    device: _DeviceType = "cpu",
    generator: Generator | None = None,
    requires_grad: bool = False,
) -> Tensor:
    sh = normalize_shape(shape)
    impl = _C_engine.normal(sh, float(mean), float(std),
                             to_engine_dtype(dtype),
                             to_engine_device(device), _gen_obj(generator))
    t = Tensor._wrap(impl)
    if requires_grad:
        t.requires_grad_(True)
    return t


def randint(
    *shape: int | _ShapeLike,
    low: int, high: int,
    dtype: _BuiltinNumeric | Numeric | None = None,
    device: _DeviceType = "cpu",
    generator: Generator | None = None,
) -> Tensor:
    sh = normalize_shape(shape)
    from lucid.types import Int64
    return Tensor._wrap(_C_engine.randint(
        sh, int(low), int(high),
        to_engine_dtype(dtype if dtype is not None else Int64),
        to_engine_device(device), _gen_obj(generator)))


def bernoulli(
    *shape: int | _ShapeLike,
    p: float,
    dtype: _BuiltinNumeric | Numeric | None = None,
    device: _DeviceType = "cpu",
    generator: Generator | None = None,
) -> Tensor:
    sh = normalize_shape(shape)
    return Tensor._wrap(_C_engine.bernoulli(
        sh, float(p), to_engine_dtype(dtype),
        to_engine_device(device), _gen_obj(generator)))


def permutation(
    n: int,
    dtype: _BuiltinNumeric | Numeric | None = None,
    device: _DeviceType = "cpu",
    generator: Generator | None = None,
) -> Tensor:
    """Return a permutation of [0, n) sampled via Fisher-Yates with the
    engine's Philox generator. Pure-engine path: arange + sort by random
    keys (so determinism follows manual_seed/Generator).
    """
    if n < 0:
        raise ValueError("permutation: n must be non-negative")
    if dtype is None:
        from lucid.types import Int64
        dtype = Int64
    # Sample float keys, sort, take indices.
    from lucid.ops.utils import argsort
    from lucid.ops.gfunc import arange
    keys = uniform(n, low=0.0, high=1.0, device=device, generator=generator)
    perm = argsort(keys, axis=0)
    if dtype is not None:
        perm = perm.astype(dtype)
    return perm


def seed(value: int) -> None:
    """Legacy alias for ``manual_seed``."""
    manual_seed(int(value))
