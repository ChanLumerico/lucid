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
    r"""Phase angle (argument) of a complex tensor.

    Computes the polar-coordinate angle of each complex element:
    :math:`\arg(z) = \arctan2(\Im(z), \Re(z))`.  The output is a real
    F32 tensor in radians on the interval :math:`[-\pi, \pi]`.

    Parameters
    ----------
    input : Tensor
        Complex-valued tensor (dtype must be ``complex64``).

    Returns
    -------
    Tensor
        Real F32 tensor with the same shape as ``input``; each element
        contains the phase of the corresponding complex entry.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = \arg(z_i) = \arctan2(\Im(z_i), \Re(z_i))

    Uses quadrant-correct :func:`lucid.atan2` internally — unlike a naive
    ``arctan(imag / real)``, this returns the proper four-quadrant angle.
    Zero complex inputs yield 0.0 (not NaN).

    Examples
    --------
    >>> import lucid
    >>> z = lucid.tensor([1+0j, 1+1j, 0+1j, -1+0j])
    >>> lucid.angle(z)
    Tensor([0.0000, 0.7854, 1.5708, 3.1416])
    """
    return lucid.atan2(lucid.imag(input), lucid.real(input))


def polar(abs: Tensor, angle: Tensor) -> Tensor:
    r"""Build a complex tensor from polar coordinates (magnitude, phase).

    Constructs each complex output entry as
    :math:`z = r e^{i\theta} = r\cos\theta + i\,r\sin\theta`, where ``r``
    comes from ``abs`` and :math:`\theta` from ``angle``.  Both inputs are
    real-valued and follow standard broadcasting rules.

    Parameters
    ----------
    abs : Tensor
        Real F32 tensor of magnitudes (non-negative values are typical
        but not enforced — a negative magnitude simply rotates the phase
        by :math:`\pi`).
    angle : Tensor
        Real F32 tensor of phase angles, in radians. Must broadcast with
        ``abs``.

    Returns
    -------
    Tensor
        ``complex64`` tensor whose shape is the broadcast shape of the two
        inputs.

    Notes
    -----
    Mathematical definition:

    .. math::

        z = r\,e^{i\theta} = r\cos\theta + i\,r\sin\theta.

    Inverse of the pair :func:`lucid.abs` / :func:`angle` for complex
    tensors: ``polar(abs(z), angle(z))`` reconstructs ``z`` up to
    floating-point round-off.

    Examples
    --------
    >>> import lucid
    >>> import math
    >>> r = lucid.tensor([1.0, 2.0])
    >>> th = lucid.tensor([0.0, math.pi / 2])
    >>> lucid.polar(r, th)
    Tensor([1.+0.j, 0.+2.j])
    """
    return lucid.complex(abs * lucid.cos(angle), abs * lucid.sin(angle))


def view_as_real(input: Tensor) -> Tensor:
    r"""Reinterpret a complex tensor as a real tensor with a trailing pair axis.

    Given an input of shape :math:`(\dots)` and dtype ``complex64``,
    returns an F32 tensor of shape :math:`(\dots, 2)` where the trailing
    axis carries ``[real, imag]`` pairs.

    Parameters
    ----------
    input : Tensor
        Complex-valued tensor (dtype ``complex64``).

    Returns
    -------
    Tensor
        Real F32 tensor of shape ``input.shape + (2,)``.

    Notes
    -----
    Conceptually this is the map

    .. math::

        z_i = a_i + i\,b_i \;\mapsto\; (a_i, b_i).

    Currently implemented as a copy via :func:`lucid.stack`, not a
    zero-copy storage reinterpret.  Apple Silicon's GPU memory model
    makes a genuine zero-copy alias tricky, and Lucid's view contract
    intentionally enforces value semantics.  Use freely for shape
    round-trips; do not rely on aliasing for in-place writes.

    Examples
    --------
    >>> import lucid
    >>> z = lucid.tensor([1+2j, 3+4j])
    >>> lucid.view_as_real(z)
    Tensor([[1., 2.],
            [3., 4.]])
    """
    return lucid.stack([lucid.real(input), lucid.imag(input)], dim=-1)


def view_as_complex(input: Tensor) -> Tensor:
    r"""Reinterpret a real tensor with a trailing pair axis as a complex tensor.

    Inverse of :func:`view_as_real`.  Given an F32 input of shape
    :math:`(\dots, 2)`, returns a ``complex64`` tensor of shape
    :math:`(\dots)` whose last axis is consumed.

    Parameters
    ----------
    input : Tensor
        Real F32 tensor whose final dimension has size 2; the two
        components are interpreted as ``(real, imag)``.

    Returns
    -------
    Tensor
        ``complex64`` tensor of shape ``input.shape[:-1]``.

    Raises
    ------
    ValueError
        If ``input.ndim < 1`` or the last axis does not have size 2.

    Notes
    -----
    Conceptually:

    .. math::

        (a_i, b_i) \;\mapsto\; z_i = a_i + i\,b_i.

    Same copy / aliasing caveat as :func:`view_as_real` — implemented as
    a materialised gather, not a zero-copy reinterpret.

    Examples
    --------
    >>> import lucid
    >>> r = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> lucid.view_as_complex(r)
    Tensor([1.+2.j, 3.+4.j])
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
