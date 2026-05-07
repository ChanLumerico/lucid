"""Complex-number composite ops: ``angle``, ``polar``, ``view_as_real``,
``view_as_complex``.

The four primitives ``real`` / ``imag`` / ``complex`` / ``conj`` are
engine ops (each backend has its own native implementation — CPU walks
the interleaved C64 storage, GPU dispatches to ``mlx::core::real`` /
``imag`` / ``conjugate``).  This module layers higher-level views on top.

* ``angle(c) = atan2(imag(c), real(c))`` — always in ``[-π, π]``.
* ``polar(abs, angle) = complex(abs * cos(angle), abs * sin(angle))``.
* ``view_as_real(c64) -> f32(..., 2)`` — interleaves real / imag
  along a new last axis.  Currently a copy (``stack``) rather than a
  zero-copy storage reinterpret — performance hot-spots can revisit.
* ``view_as_complex(f32(..., 2)) -> c64(...)`` — inverse of the above.
"""

from typing import TYPE_CHECKING

import lucid

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def angle(input: Tensor) -> Tensor:
    """Phase angle (argument) of a complex tensor — output is real F32 in
    radians.  ``input`` must be C64."""
    return lucid.atan2(lucid.imag(input), lucid.real(input))


def polar(abs: Tensor, angle: Tensor) -> Tensor:
    """Build a complex tensor from magnitude / phase: ``|abs| · e^{i·angle}``.

    Both inputs are real (F32) and broadcast-compatible; the output is C64
    with shape equal to the broadcast shape of the two inputs.
    """
    return lucid.complex(abs * lucid.cos(angle), abs * lucid.sin(angle))


def view_as_real(input: Tensor) -> Tensor:
    """Reinterpret a C64 tensor of shape ``(...,)`` as an F32 tensor of
    shape ``(..., 2)`` where the last axis holds ``[real, imag]``.

    .. note::
       Currently a copy — Apple Silicon's GPU memory model makes a
       genuine zero-copy storage reinterpret tricky, and the reference
       framework's view contract demands aliasing semantics.  The copy
       here is correct under value semantics; use it freely for shape
       round-trips, not for in-place writes.
    """
    return lucid.stack([lucid.real(input), lucid.imag(input)], dim=-1)


def view_as_complex(input: Tensor) -> Tensor:
    """Inverse of :func:`view_as_real`: a real F32 tensor of shape
    ``(..., 2)`` becomes a C64 tensor of shape ``(...,)`` where the last
    axis is consumed.

    Same copy / aliasing caveat as :func:`view_as_real`.
    """
    if input.ndim < 1 or int(input.shape[-1]) != 2:
        raise ValueError(
            "view_as_complex: expected last dim == 2, got "
            f"shape {tuple(input.shape)}"
        )
    re = lucid.gather(
        input,
        lucid.zeros_like(input, dtype=lucid.int32)[..., :1],
        dim=-1,
    ).squeeze(-1)
    im = lucid.gather(
        input,
        lucid.zeros_like(input, dtype=lucid.int32)[..., :1] + 1,
        dim=-1,
    ).squeeze(-1)
    return lucid.complex(re, im)


__all__ = [
    "angle",
    "polar",
    "view_as_real",
    "view_as_complex",
]
