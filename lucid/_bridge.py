"""
lucid._bridge — private Python ↔ C++ engine conversion utilities.

Every wrapper in `lucid.ops`, `lucid.nn.functional`, etc. uses these to
normalize Python-flavored arguments (built-in types, lucid.types.Numeric,
numpy dtypes, etc.) into the C++ engine's strict enums (`Dtype`, `Device`)
and shape representations.

Not part of the public API — `_` prefix.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from lucid._C import engine as _C_engine
from lucid.error import UnknownDeviceError
from lucid.types import (
    Numeric,
    _DeviceType,
    _ShapeLike,
    Float32,
    to_numeric_type,
)


# --------------------------------------------------------------------------- #
# Dtype
# --------------------------------------------------------------------------- #

# Reverse table: engine Dtype → Numeric (cached on first access).
_ENGINE_TO_NUMERIC: dict["_C_engine.Dtype", Numeric] | None = None


def _build_engine_to_numeric_table() -> dict["_C_engine.Dtype", Numeric]:
    from lucid.types import (
        Bool, Int8, Int16, Int32, Int64,
        Float16, Float32, Float64, Complex64,
    )
    return {
        _C_engine.Dtype.Bool: Numeric(int, 8),  # Bool → smallest int placeholder
        _C_engine.Dtype.I8:   Int8,
        _C_engine.Dtype.I16:  Int16,
        _C_engine.Dtype.I32:  Int32,
        _C_engine.Dtype.I64:  Int64,
        _C_engine.Dtype.F16:  Float16,
        _C_engine.Dtype.F32:  Float32,
        _C_engine.Dtype.F64:  Float64,
        _C_engine.Dtype.C64:  Complex64,
    }


def to_engine_dtype(dt: Any, *, default: "_C_engine.Dtype" = _C_engine.Dtype.F32
                    ) -> "_C_engine.Dtype":
    """Coerce a Python-flavored dtype handle to a C++ engine `Dtype`.

    Accepted inputs:
      - `None`              → `default` (F32 by default)
      - `_C_engine.Dtype`   → returned as-is
      - `lucid.types.Numeric`  → `.engine_dtype` (raises if bit-free)
      - numpy dtype / type     → resolved via `to_numeric_type`
      - Python `int`, `float`, `complex`, `bool` → I64, F32, C64, Bool
      - string ('float32', 'int64', etc.) → resolved via numpy
    """
    if dt is None:
        return default
    if isinstance(dt, _C_engine.Dtype):
        return dt
    if isinstance(dt, Numeric):
        eng = dt.engine_dtype
        if eng is None:
            # Bit-free Numeric (Float, Int, ...) — fall back to default sized.
            if dt.base_dtype is float:    return _C_engine.Dtype.F32
            if dt.base_dtype is int:      return _C_engine.Dtype.I64
            if dt.base_dtype is complex:  return _C_engine.Dtype.C64
            return default
        return eng

    # Python builtin types
    if dt is bool:    return _C_engine.Dtype.Bool
    if dt is int:     return _C_engine.Dtype.I64
    if dt is float:   return _C_engine.Dtype.F32
    if dt is complex: return _C_engine.Dtype.C64

    # numpy dtype / type / string — go via to_numeric_type then map
    try:
        num = to_numeric_type(dt)
        eng = num.engine_dtype
        if eng is not None:
            return eng
    except (KeyError, IndexError, TypeError):
        pass

    raise TypeError(f"Cannot convert {dt!r} to engine Dtype")


def from_engine_dtype(eng_dt: "_C_engine.Dtype") -> Numeric:
    """Reverse of `to_engine_dtype` — engine Dtype → user-visible Numeric."""
    global _ENGINE_TO_NUMERIC
    if _ENGINE_TO_NUMERIC is None:
        _ENGINE_TO_NUMERIC = _build_engine_to_numeric_table()
    return _ENGINE_TO_NUMERIC[eng_dt]


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #

def to_engine_device(device: Any,
                     *, default: "_C_engine.Device" = _C_engine.Device.CPU
                     ) -> "_C_engine.Device":
    """Coerce a Python device representation to `_C_engine.Device`.

    Accepts: None (→ default), 'cpu'/'gpu' strings, `_C_engine.Device`.
    """
    if device is None:
        return default
    if isinstance(device, _C_engine.Device):
        return device
    if isinstance(device, str):
        d = device.lower()
        if d == "cpu": return _C_engine.Device.CPU
        if d == "gpu": return _C_engine.Device.GPU
        raise UnknownDeviceError(device)
    raise UnknownDeviceError(str(device))


def from_engine_device(eng_dev: "_C_engine.Device") -> _DeviceType:
    """Engine Device → 'cpu' | 'gpu' string."""
    if eng_dev == _C_engine.Device.CPU: return "cpu"
    if eng_dev == _C_engine.Device.GPU: return "gpu"
    raise UnknownDeviceError(str(eng_dev))


# --------------------------------------------------------------------------- #
# Shape
# --------------------------------------------------------------------------- #

def normalize_shape(args: Iterable[Any] | int | _ShapeLike) -> list[int]:
    """Resolve overloaded shape arguments to a flat list of ints.

    Examples:
      normalize_shape((3, 4))          → [3, 4]
      normalize_shape([3, 4])          → [3, 4]
      normalize_shape(((3, 4),))       → [3, 4]   (tuple wrapping)
      normalize_shape(([3, 4],))       → [3, 4]   (list wrapping)
      normalize_shape(3)               → [3]
    """
    # Single int
    if isinstance(args, int):
        return [args]
    # Direct sequence (list/tuple of ints)
    if isinstance(args, (list, tuple)) and len(args) > 0:
        first = args[0]
        if isinstance(first, (list, tuple)):
            # Single-element wrapper containing the actual shape
            if len(args) == 1:
                return [int(d) for d in first]
            # Otherwise: tuple of mixed; flatten any inner tuple
            out: list[int] = []
            for a in args:
                if isinstance(a, (list, tuple)):
                    out.extend(int(d) for d in a)
                else:
                    out.append(int(a))
            return out
        # Sequence of ints
        return [int(d) for d in args]
    if isinstance(args, (list, tuple)) and len(args) == 0:
        return []
    # Iterable fallback
    if hasattr(args, "__iter__"):
        return [int(d) for d in args]
    raise TypeError(f"Cannot normalize shape from {args!r}")


# --------------------------------------------------------------------------- #
# Wrap / unwrap helpers (used pervasively by ops/* and nn/*)
# --------------------------------------------------------------------------- #

def impl_of(t: Any) -> "_C_engine.TensorImpl":
    """Extract the underlying TensorImpl from a lucid.Tensor (lazy import)."""
    # Lazy import to avoid circularity at module load time.
    from lucid._tensor import Tensor
    if isinstance(t, Tensor):
        return t._impl
    if isinstance(t, _C_engine.TensorImpl):
        return t
    raise TypeError(f"Expected lucid.Tensor or TensorImpl, got {type(t).__name__}")


def impls_of(ts: Sequence[Any]) -> list["_C_engine.TensorImpl"]:
    """Vectorized `impl_of`."""
    return [impl_of(t) for t in ts]
