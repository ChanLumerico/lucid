"""
lucid.types — public type aliases, the `Numeric` dtype handle, and the
runtime-checkable `_TensorLike` / `_ModuleHookable` protocols used by the
hook system.

The new lucid Python layer is a thin wrapper over the C++ engine
(`lucid._C.engine`). To keep models and external code unchanged, every
`Numeric` instance also exposes a `.cpu`/`.gpu` view that resolves to the
corresponding numpy/mlx dtype — this is what user code (and the engine's
`from_numpy` bridge) reads when it needs a backend-native dtype handle.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Literal,
    Protocol,
    Self,
    Sequence,
    TypeAlias,
    runtime_checkable,
)

import numpy as np
import mlx.core as mx

from lucid._C import engine as _C_engine


# --------------------------------------------------------------------------- #
# Scalar / array / shape aliases
# --------------------------------------------------------------------------- #

_DeviceType = Literal["cpu", "gpu"]

_Scalar = int | float | complex
_NumPyArray: TypeAlias = np.ndarray
_MLXArray: TypeAlias = mx.array

_TensorData = _NumPyArray | _MLXArray
_Gradient = _NumPyArray | _MLXArray | None
_ArrayOrScalar = _Scalar | list[_Scalar] | _NumPyArray | _MLXArray

_BuiltinNumeric = type[bool | int | float | complex]

_ShapeLike = list[int] | tuple[int, ...]

_IndexLike = int | slice | Sequence[int]

_ArrayLike = list | _NumPyArray | _MLXArray
_ArrayLikeInt = int | Sequence[int | tuple[int, int]]

_OptimClosure = Callable[[], Any]

_EinopsPattern = str


# --------------------------------------------------------------------------- #
# `_TensorLike` / `_ModuleHookable` — runtime-checkable protocols
# --------------------------------------------------------------------------- #

@runtime_checkable
class _TensorLike(Protocol):
    dtype: Any
    requires_grad: bool
    is_free: bool
    device: _DeviceType
    shape: Any
    data: Any
    grad: Any
    keep_grad: bool
    is_leaf: bool
    size: Any

    _op: object | None
    _prev: list[_TensorLike]
    _backward_op: Any
    _backward_hooks: Any
    _version: int

    def to(self, device: _DeviceType) -> None: ...
    def free(self) -> None: ...
    def new_tensor(self) -> _TensorLike: ...
    def is_cpu(self) -> bool: ...
    def is_gpu(self) -> bool: ...
    def clear_node(self, clear_op: bool = True) -> None: ...
    def backward(
        self, retain_grad: bool = False, retain_graph: bool = False
    ) -> None: ...


@runtime_checkable
class _ModuleHookable(Protocol):
    def register_forward_pre_hook(
        self, hook: Callable, *, with_kwargs: bool = False
    ) -> Callable: ...

    def register_forward_hook(
        self, hook: Callable, *, with_kwargs: bool = False
    ) -> Callable: ...

    def register_backward_hook(self, hook: Callable) -> Callable: ...
    def register_full_backward_pre_hook(self, hook: Callable) -> Callable: ...
    def register_full_backward_hook(self, hook: Callable) -> Callable: ...
    def register_state_dict_pre_hook(self, hook: Callable) -> Callable: ...
    def register_state_dict_hook(self, hook: Callable) -> Callable: ...
    def register_load_state_dict_pre_hook(self, hook: Callable) -> Callable: ...
    def register_load_state_dict_post_hook(self, hook: Callable) -> Callable: ...


# Hook function signatures (typing only)
_ForwardPreHook: TypeAlias = Callable[
    [_ModuleHookable, tuple[Any, ...]], tuple[Any, ...] | None
]
_ForwardPreHookKwargs: TypeAlias = Callable[
    [_ModuleHookable, tuple[Any, ...], dict[str, Any]],
    tuple[tuple[Any, ...], dict[str, Any]] | None,
]
_ForwardHook: TypeAlias = Callable[
    [_ModuleHookable, tuple[Any, ...], Any], Any | None
]
_ForwardHookKwargs: TypeAlias = Callable[
    [_ModuleHookable, tuple[Any, ...], dict[str, Any], Any], Any | None
]

_BackwardHook: TypeAlias = Callable[[_TensorLike, _NumPyArray], None]
_FullBackwardPreHook: TypeAlias = Callable[
    [_ModuleHookable, tuple[_NumPyArray | None, ...]],
    tuple[_NumPyArray | None, ...] | None,
]
_FullBackwardHook: TypeAlias = Callable[
    [_ModuleHookable,
     tuple[_NumPyArray | None, ...],
     tuple[_NumPyArray | None, ...]],
    tuple[_NumPyArray | None, ...] | None,
]

_StateDictPreHook: TypeAlias = Callable[[_ModuleHookable, str, bool], None]
_StateDictHook: TypeAlias = Callable[
    [_ModuleHookable, OrderedDict, str, bool], None
]
_LoadStateDictPreHook: TypeAlias = Callable[
    [_ModuleHookable, OrderedDict, bool], None
]
_LoadStateDictPostHook: TypeAlias = Callable[
    [_ModuleHookable, set[str], set[str], bool], None
]


# --------------------------------------------------------------------------- #
# Numeric — user-facing dtype handle
# --------------------------------------------------------------------------- #

class Numeric:
    """A symbolic dtype handle that bridges Python type → numpy/mlx dtype.

    `Numeric(int, 32)` represents int32. `.cpu` returns `np.int32`, `.gpu`
    returns `mx.int32`. `.engine_dtype` returns the matching
    `lucid._C.engine.Dtype` enum (preferred for new code that talks
    directly to the C++ engine).

    Bit-free instances (`Numeric(int, None)`) are wildcards used for type
    coercion: they equal any same-base-type sized counterpart.
    """

    def __init__(
        self, base_dtype: type[int | float | complex], bits: int | None
    ) -> None:
        self.base_dtype = base_dtype
        self.base_str = base_dtype.__name__
        self.bits = bits

        self._np_dtype: type | None = None
        self._mlx_dtype: type | None = None

        if bits is not None:
            self._np_dtype = getattr(np, self.base_str + str(bits))
            bits_mlx = bits
            if (
                mx.default_device().type is mx.DeviceType.gpu
                and self.base_dtype is float
                and bits == 64
            ):
                bits_mlx = 32
            self._mlx_dtype = getattr(mx, self.base_str + str(bits_mlx))

    # -- numpy / mlx views ---------------------------------------------------
    @property
    def cpu(self) -> type | None:
        return self._np_dtype

    @property
    def gpu(self) -> type | None:
        return self._mlx_dtype

    @property
    def is_bit_free(self) -> bool:
        return self.bits is None

    # -- C++ engine view -----------------------------------------------------
    @property
    def engine_dtype(self) -> "_C_engine.Dtype | None":
        if self.bits is None:
            return None
        return _NUMERIC_TO_ENGINE.get((self.base_dtype, self.bits))

    # -- helpers -------------------------------------------------------------
    def parse(self, device: _DeviceType) -> type | None:
        return self.cpu if device == "cpu" else self.gpu

    def auto_parse(self, data_dtype: type, device: _DeviceType) -> type | None:
        bits = self._dtype_bits(data_dtype)
        new_dtype = self.base_dtype.__name__ + str(bits)
        return getattr(np if device == "cpu" else mx, new_dtype, None)

    def _dtype_bits(self, dtype: type) -> int:
        if isinstance(dtype, (np.dtype, type)) and hasattr(dtype, "itemsize"):
            return np.dtype(dtype).itemsize * 8
        if isinstance(dtype, (mx.Dtype, type)):
            return dtype.size * 8
        if isinstance(dtype, str):
            try:
                return np.dtype(dtype).itemsize * 8
            except TypeError:
                return dtype.size * 8
        raise TypeError(f"Unsupported dtype: {dtype}")

    # -- equality / hashing --------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Numeric):
            return NotImplemented
        if self.base_dtype is not other.base_dtype:
            return False
        if self.is_bit_free:
            return other.is_bit_free
        return other.is_bit_free or self.bits == other.bits

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return self.base_dtype.__name__ + (str(self.bits) if self.bits else "")

    def __repr__(self) -> str:
        return (
            f"Numeric(base_dtype={self.base_dtype.__name__}, bits={self.bits},"
            f" cpu={self.cpu}, gpu={self.gpu})"
        )


# Singletons -- imported by users as `lucid.types.Float32`, `lucid.Float32`, etc.
Int = Numeric(int, None)
Int8 = Numeric(int, bits=8)
Int16 = Numeric(int, bits=16)
Int32 = Numeric(int, bits=32)
Int64 = Numeric(int, bits=64)

Char = Int8
Short = Int16
Long = Int64

Float = Numeric(float, None)
Float16 = Numeric(float, bits=16)
Float32 = Numeric(float, bits=32)
Float64 = Numeric(float, bits=64)

Half = Float16
Double = Float64

Complex = Numeric(complex, None)
Complex64 = Numeric(complex, bits=64)

# Boolean placeholder (bits=8, base int) — engine_dtype maps to Dtype.Bool.
Bool = Numeric(int, 8)


# --------------------------------------------------------------------------- #
# Numeric ↔ C++ engine `Dtype` mapping
# --------------------------------------------------------------------------- #

_NUMERIC_TO_ENGINE: dict[tuple[type, int], "_C_engine.Dtype"] = {
    (int,     8):  _C_engine.Dtype.I8,
    (int,    16):  _C_engine.Dtype.I16,
    (int,    32):  _C_engine.Dtype.I32,
    (int,    64):  _C_engine.Dtype.I64,
    (float,  16):  _C_engine.Dtype.F16,
    (float,  32):  _C_engine.Dtype.F32,
    (float,  64):  _C_engine.Dtype.F64,
    (complex, 64): _C_engine.Dtype.C64,
}


# --------------------------------------------------------------------------- #
# Lookup tables / conversion helpers
# --------------------------------------------------------------------------- #

numeric_dict: dict[str, dict[str, Numeric]] = {
    "int":     {"8": Int8, "16": Int16, "32": Int32, "64": Int64},
    "float":   {"16": Float16, "32": Float32, "64": Float64},
    "complex": {"64": Complex64},
}


def to_numeric_type(data_dtype: type) -> Numeric:
    """Resolve a numpy/mlx/string dtype representation to a `Numeric`."""
    str_dtype = str(data_dtype).split(".")[-1]
    name = re.findall(r"[a-z]+", str_dtype)[0]
    bits = re.findall(r"\d+", str_dtype)[0]
    return numeric_dict[name][bits]


__all__ = [
    # Aliases
    "_DeviceType", "_Scalar", "_NumPyArray", "_MLXArray", "_TensorData",
    "_Gradient", "_ArrayOrScalar", "_BuiltinNumeric", "_ShapeLike",
    "_IndexLike", "_ArrayLike", "_ArrayLikeInt", "_OptimClosure",
    "_EinopsPattern",
    # Protocols
    "_TensorLike", "_ModuleHookable",
    # Hook signatures
    "_ForwardPreHook", "_ForwardPreHookKwargs", "_ForwardHook",
    "_ForwardHookKwargs", "_BackwardHook", "_FullBackwardPreHook",
    "_FullBackwardHook", "_StateDictPreHook", "_StateDictHook",
    "_LoadStateDictPreHook", "_LoadStateDictPostHook",
    # Numeric class & singletons
    "Numeric",
    "Int", "Int8", "Int16", "Int32", "Int64",
    "Char", "Short", "Long",
    "Float", "Float16", "Float32", "Float64",
    "Half", "Double",
    "Complex", "Complex64",
    "Bool",
    # Helpers
    "numeric_dict", "to_numeric_type",
]
