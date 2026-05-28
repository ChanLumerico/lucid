"""Tiny helpers shared across the distribution implementations."""

import lucid
from lucid._tensor.tensor import Tensor


def _as_tensor(x: Tensor | float | int) -> Tensor:
    """Promote a Python scalar to a 0-dim Lucid tensor — matches the
    reference framework's convention so distributions parameterised by
    scalars get ``batch_shape == ()`` rather than ``(1,)``."""
    if isinstance(x, lucid.Tensor):
        return x
    t = lucid.tensor(float(x))
    if t.ndim == 1 and t.shape == (1,):
        t = t.squeeze()
    return t


def _broadcast_pair(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Broadcast ``a`` and ``b`` to a common shape via arithmetic ``+ 0``.

    Cheaper than constructing two ``broadcast_to`` views — the
    ``a*0 + b*0`` trick produces a single materialised zero tensor of
    the joint shape, then both inputs are added to it.  Returns
    ``(a, b)`` unchanged when shapes already match (no zero
    materialisation).

    Parameters
    ----------
    a, b : Tensor
        Inputs to align.  Must be broadcast-compatible per the standard
        rules; otherwise the underlying engine raises.

    Returns
    -------
    tuple of Tensor
        ``(a_broadcast, b_broadcast)`` with identical shapes.
    """
    if tuple(a.shape) == tuple(b.shape):
        return a, b
    z = a * 0 + b * 0
    return a + z, b + z
