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
Internal modules import the aliases they need directly::

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
    ParamSpec,
    Protocol,
    TypedDict,
    TypeVar,
    runtime_checkable,
)

import numpy as np

from lucid._tensor.tensor import Tensor
from lucid._dtype import dtype as _DType
from lucid._device import device as _Device

# ── TypeVars ──────────────────────────────────────────────────────────────────

DT = TypeVar("DT", bound=_DType)
"""TypeVar for dtype-parametric functions and return-type preservation."""

DV = TypeVar("DV", bound=_Device)
"""TypeVar for device-parametric functions and return-type preservation."""

_T = TypeVar("_T")
"""Generic return TypeVar used in decorator helpers."""

_P = ParamSpec("_P")
"""ParamSpec for preserving callable signatures through decorators."""

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


# ── Public type aliases ───────────────────────────────────────────────────────

# Scalar numeric types — valid operands alongside Tensor in arithmetic.
type Scalar = int | float | bool

# Any value that lucid can meaningfully convert to a Tensor.
type TensorLike = Tensor | np.ndarray | list[object] | int | float | bool

# Device specifier: an actual device object, a string name ('cpu'/'metal'), or None.
type DeviceLike = _Device | str | None

# DType specifier: an actual dtype object or None (→ use the global default).
type DTypeLike = _DType | None

# Shape / size specifier used in factory functions and reshape.
type ShapeLike = int | tuple[int, ...]

# ── Shape variants for spatial 1-D / 2-D / 3-D ops ──────────────────────────
# Used in Conv, Pool, and similar spatial layers.

# Single spatial dimension (Conv1d, MaxPool1d, …).
type _Size1d = int | tuple[int]

# Two spatial dimensions (Conv2d, MaxPool2d, …).
type _Size2d = int | tuple[int, int]

# Three spatial dimensions (Conv3d, MaxPool3d, …).
type _Size3d = int | tuple[int, int, int]

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
