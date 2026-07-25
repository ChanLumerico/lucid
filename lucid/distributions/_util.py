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
    # Distribution parameters routinely mix a user tensor with a host-derived
    # scalar constant (``Chi2`` passes ``rate=_as_tensor(0.5)``, ``StudentT``
    # its ``loc``/``scale`` defaults).  Those constants are created without a
    # device, so combining them with a Metal parameter raised DeviceMismatch.
    # This is the single point where the two meet, so reconcile here rather
    # than threading a device through all ~67 ``_as_tensor`` call sites.
    if a.device != b.device:
        if a.numel() == 1 and b.numel() != 1:
            a = a.to(b.device.type)
        elif b.numel() == 1 and a.numel() != 1:
            b = b.to(a.device.type)
        else:
            # Neither is a lone scalar: prefer the non-CPU side so the
            # distribution keeps living where the user put it.
            target = a.device if a.device.type != "cpu" else b.device
            a = a.to(target.type)
            b = b.to(target.type)
    if tuple(a.shape) == tuple(b.shape):
        return a, b
    z = a * 0 + b * 0
    return a + z, b + z
