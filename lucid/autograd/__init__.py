"""
lucid.autograd — public autograd surface over the C++ engine.

The engine builds and traverses the autograd graph internally; this
module just exposes the user-facing entry points (`backward`, `grad`,
`no_grad`, ...). They forward to `_C_engine.engine_backward` and
`_C_engine.{grad_enabled, set_grad_enabled, NoGradGuard}`.
"""

from __future__ import annotations

from lucid.autograd.function import (
    backward,
    grad,
    no_grad,
    enable_grad,
    set_grad_enabled,
    is_grad_enabled,
)


__all__ = [
    "backward",
    "grad",
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "is_grad_enabled",
]
