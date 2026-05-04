"""
lucid._types: canonical type aliases, protocols, and TypeVars for the lucid API.

These serve three purposes:
  1. **Readability** — ``def to(self, device: DeviceLike)`` is clearer than
     a raw union.
  2. **DRY** — each underlying union is written once and referenced everywhere.
  3. **Extensibility** — :class:`TensorLikeProtocol` lets user-defined types
     participate in lucid's functional interface without subclassing ``Tensor``.

Import style
------------
Early modules (factories, dispatch, _dtype, _device) that are themselves imported
by ``tensor.py`` must use :mod:`lucid._types_base` to avoid circular imports.
All other modules may import from here directly::

    from lucid._types import DeviceLike, DTypeLike, ShapeLike, _ModuleOutput

Public-facing names (``Scalar``, ``TensorLike``, ``DeviceLike``, …) are also
re-exported from ``lucid.__init__`` for advanced users.

Naming conventions
------------------
- No leading underscore → public API (stable).
- Leading underscore   → private / implementation detail (may change).
"""

from typing import (
    Callable,
    Protocol,
    TypedDict,
    runtime_checkable,
)

import numpy as np

from lucid._tensor.tensor import Tensor
from lucid._dtype import dtype as _DType
from lucid._device import device as _Device

# ── Re-export everything from _types_base ─────────────────────────────────────
# Consumers that only need the Tensor-free aliases can import from _types_base;
# everyone else can import from here and get the full set.
from lucid._types_base import (  # noqa: F401
    DT, DV, _T, _P,
    Scalar, DeviceLike, DTypeLike, ShapeLike,
    _Size1d, _Size2d, _Size3d,
)

# ── Protocols ─────────────────────────────────────────────────────────────────


@runtime_checkable
class HasShape(Protocol):
    """Structural protocol: anything with ``.shape`` and ``.ndim``."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...


@runtime_checkable
class SupportsNumpyConversion(Protocol):
    """Structural protocol: anything convertible to a :class:`numpy.ndarray`."""

    def numpy(self) -> np.ndarray: ...  # type: ignore[type-arg]


@runtime_checkable
class SupportsGrad(Protocol):
    """Structural protocol: anything that participates in autograd."""

    @property
    def requires_grad(self) -> bool: ...

    def backward(self, *, retain_graph: bool = ...) -> None: ...

    def detach(self) -> "SupportsGrad": ...


@runtime_checkable
class TensorLikeProtocol(HasShape, SupportsNumpyConversion, Protocol):
    """Structural protocol for objects that behave like :class:`Tensor`.

    Any object satisfying this protocol can be used where a Tensor is expected
    in lucid's functional interfaces — enabling custom array types without
    subclassing.

    Example
    -------
    >>> class MyArray:
    ...     @property
    ...     def shape(self) -> tuple[int, ...]: return (3,)
    ...     @property
    ...     def ndim(self) -> int: return 1
    ...     @property
    ...     def dtype(self) -> lucid.dtype: return lucid.float32
    ...     @property
    ...     def device(self) -> lucid.device: return lucid.device("cpu")
    ...     def numpy(self): return np.zeros(3)
    ...     def to(self, *a, **kw): return self
    >>> isinstance(MyArray(), TensorLikeProtocol)
    True
    """

    @property
    def dtype(self) -> _DType: ...

    @property
    def device(self) -> _Device: ...

    def to(self, *args: object, **kwargs: object) -> "TensorLikeProtocol": ...


# ── TypedDict: optimizer parameter group ─────────────────────────────────────


class ParamGroupDict(TypedDict, total=False):
    """TypedDict for optimizer parameter groups.

    ``params`` is the only required key; all other fields are optimizer-specific.
    """

    params: list[object]  # list[Parameter] at runtime
    lr: float
    weight_decay: float
    momentum: float
    dampening: float
    betas: tuple[float, float]
    eps: float
    amsgrad: bool
    nesterov: bool
    maximize: bool
    foreach: bool | None
    fused: bool | None


# ── Public type aliases (Tensor-dependent) ────────────────────────────────────

# Any value that lucid can meaningfully convert to a Tensor.
type TensorLike = Tensor | np.ndarray | list[object] | int | float | bool

# ── Tensor operand / indexing ─────────────────────────────────────────────────

# Right-hand operand for arithmetic dunders (+, -, *, /, **, …).
type TensorOrScalar = Tensor | Scalar

# All legal index forms for Tensor.__getitem__ / __setitem__.
type _IndexType = (
    int
    | slice
    | Tensor
    | list[int]
    | tuple[int | slice | Tensor | list[int] | None, ...]
    | None
)

# ── Neural network ────────────────────────────────────────────────────────────

# Return type of Module.forward: either a single Tensor or a tuple of Tensors.
type _ModuleOutput = Tensor | tuple[Tensor, ...]

# State dict: flat mapping from dotted parameter / buffer name to its tensor.
type StateDict = dict[str, Tensor]

# Hook signatures for Module.register_forward_pre_hook / register_forward_hook.
# Module is typed as object here to avoid a circular import (module.py → _types.py → module.py).
type _ForwardPreHook = Callable[[object, tuple[Tensor, ...]], tuple[Tensor, ...] | None]
type _ForwardHook = Callable[[object, tuple[Tensor, ...], _ModuleOutput], _ModuleOutput | None]
type _BackwardHook = Callable[..., None]

# ── Optimizers ────────────────────────────────────────────────────────────────

# A single parameter group dict holding hyperparams and a list of Parameters.
type _ParamGroup = dict[str, object]

# Optional re-evaluation closure passed to Optimizer.step().
type _OptimizerClosure = Callable[[], Tensor] | None
