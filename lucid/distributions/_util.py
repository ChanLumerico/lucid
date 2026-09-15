"""Tiny helpers shared across the distribution implementations."""

from typing import Any, Callable, Self, overload

import lucid
from lucid._dtype import finfo
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


def _broadcast_shapes(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """The right-aligned broadcast of two shapes, as a shape.

    Wanted where the two things being aligned are not both tensors — a
    ``log_prob`` has to reconcile a *value's* shape with the
    distribution's ``batch_shape``, and ``broadcast_to`` alone only ever
    grows one side, so whichever side happened to be shorter decided
    which calls worked.

    Parameters
    ----------
    a, b : tuple of int
        Shapes to align.

    Returns
    -------
    tuple of int
        The joint shape.

    Raises
    ------
    ValueError
        If a dimension pair is neither equal nor has a ``1`` in it.
    """
    out: list[int] = []
    for i in range(max(len(a), len(b))):
        da = a[len(a) - 1 - i] if i < len(a) else 1
        db = b[len(b) - 1 - i] if i < len(b) else 1
        if da != db and da != 1 and db != 1:
            raise ValueError(f"shapes {a} and {b} are not broadcast-compatible")
        out.append(da if db == 1 else db)
    out.reverse()
    return tuple(out)


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


def _clamp_probs(probs: Tensor) -> Tensor:
    """``probs`` held one machine epsilon inside ``[0, 1]``.

    What the reference framework does before the logarithm that turns
    probabilities into logits: an exact 0 or 1 would otherwise give an
    infinite logit, and a derived ``logits`` attribute is expected to be
    finite whatever ``probs`` it came from.
    """
    eps = float(finfo(probs.dtype).eps)
    return probs.clip(eps, 1.0 - eps)


class _lazy_param:
    """A parameter a distribution either stores or derives from its dual.

    ``Bernoulli`` takes ``probs`` *or* ``logits``, and the reference
    framework answers both attributes whichever one was given.  Assigning
    stores the value on the instance; reading an attribute that was never
    assigned derives it from the other one.

    The derivation runs on every read rather than once.  A cached value
    goes stale the moment the stored parameter is trained in place, and it
    ties every later use to the autograd graph of whichever step happened
    to read it first.

    Parameters
    ----------
    derive : callable
        ``derive(dist)`` computes the value from the stored dual.
    """

    def __init__(self, derive: Callable[[Any], Tensor]) -> None:
        self._derive = derive
        self._name = derive.__name__
        self.__doc__ = derive.__doc__

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    @overload
    def __get__(self, instance: None, owner: type | None = None) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type | None = None) -> Tensor: ...

    def __get__(
        self, instance: object | None, owner: type | None = None
    ) -> Self | Tensor:
        if instance is None:
            return self
        stored: Tensor | None = vars(instance).get(self._name)
        return stored if stored is not None else self._derive(instance)

    def __set__(self, instance: object, value: Tensor) -> None:
        vars(instance)[self._name] = value

    def is_stored(self, instance: object) -> bool:
        """Whether ``instance`` was given this parameter rather than its dual."""
        return self._name in vars(instance)
