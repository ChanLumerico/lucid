"""The graph node an in-place overwrite leaves behind.

``y.fill_(v)`` and ``y.zero_()`` replace every value of ``y``, so what ``y``
holds afterwards no longer depends on what it held before.  The reference
framework keeps ``y`` in the graph anyway, with that derivative — zero — so
a loss built from it can still be differentiated and sends zeros upstream.
Lucid detached it instead, which made the same loss impossible to
differentiate once ``backward()`` started refusing roots with no graph.

The node is a Python :class:`~lucid.autograd.function.Function` because it
saves nothing: an engine op applied to ``y`` records ``y``'s version, and the
assignment that follows bumps it, so its backward would report an in-place
modification that the node does not care about.
"""

from typing import final, override

from lucid._factories.creation import zeros_like
from lucid._tensor.tensor import Tensor
from lucid.autograd.function import Function, FunctionCtx


@final
class _Overwritten(Function):
    """``new``'s values, with a zero derivative with respect to ``old``."""

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, old: Tensor, new: Tensor) -> Tensor:
        return new.detach()

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[Tensor, None]:
        return zeros_like(grad), None


def overwritten(old: Tensor, new: Tensor) -> Tensor:
    """``new``'s values, standing in for ``old`` in the graph.

    Parameters
    ----------
    old : Tensor
        The tensor being overwritten; it requires grad and has a graph.
    new : Tensor
        The values written over it, which depend on nothing.

    Returns
    -------
    Tensor
        ``new``'s values with a zero derivative with respect to ``old``.
    """
    out = _Overwritten.apply(old, new)
    assert isinstance(out, Tensor)
    return out
