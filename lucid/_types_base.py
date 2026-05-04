"""
lucid._types_base: type aliases and protocols that do NOT depend on Tensor.

This module exists to break the circular import between ``_types.py``
(which imports ``Tensor``) and modules such as ``_factories/`` (which are
imported by ``tensor.py`` itself).

Import style
------------
- Modules that are imported early (factories, dispatch, dtype, device) should
  import from here.
- Modules that are imported after ``Tensor`` is defined can import from the
  richer ``lucid._types``, which re-exports everything from this module.

Naming conventions
------------------
- No leading underscore → public API (stable).
- Leading underscore   → private / implementation detail (may change).
"""

from typing import ParamSpec, TypeVar

from lucid._dtype import dtype as _DType
from lucid._device import device as _Device

# ── TypeVars ──────────────────────────────────────────────────────────────────

DT = TypeVar("DT", bound=_DType)
"""TypeVar for dtype-parametric functions."""

DV = TypeVar("DV", bound=_Device)
"""TypeVar for device-parametric functions."""

_T = TypeVar("_T")
"""Generic return TypeVar used in decorator helpers."""

_P = ParamSpec("_P")
"""ParamSpec for preserving callable signatures through decorators."""

# ── Public type aliases (no Tensor dependency) ────────────────────────────────

# Scalar numeric types — valid operands alongside Tensor in arithmetic.
type Scalar = int | float | bool

# Device specifier: an actual device object, a string name ('cpu'/'metal'), or None.
type DeviceLike = _Device | str | None

# DType specifier: an actual dtype object or None (→ use the global default).
type DTypeLike = _DType | None

# Shape / size specifier used in factory functions and reshape.
type ShapeLike = int | tuple[int, ...]

# ── Shape variants for spatial 1-D / 2-D / 3-D ops ──────────────────────────
# Used in Conv, Pool, and similar spatial layers.

type _Size1d = int | tuple[int]
type _Size2d = int | tuple[int, int]
type _Size3d = int | tuple[int, int, int]
