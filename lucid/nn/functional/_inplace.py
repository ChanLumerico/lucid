"""The one route ``inplace=True`` takes through ``nn.functional``.

Every activation and dropout that takes ``inplace`` used to accept the flag
and return a fresh tensor, leaving ``x`` as it was: ``F.relu(x, inplace=True)``
and ``nn.ReLU(inplace=True)`` computed the right values into an object the
caller did not hold.  Each one now computes its out-of-place result and hands
it to :func:`_in_place`, so they all refuse, record and write through views
the same way.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from lucid._ops.composite.indexing import _write_inplace
from lucid._tensor._metal_views import _check as _metal_check
from lucid._tensor._metal_views import write_back as _metal_write_back

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _in_place(x: Tensor, name: str, out_of_place: Callable[[Tensor], Tensor]) -> Tensor:
    """Make ``out_of_place(x)`` the values of ``x``, as an in-place op does, and return ``x``.

    * A leaf that requires grad is refused while autograd records — and so
      is a view of one — before anything is written, as the reference
      refuses it.  Under :func:`lucid.no_grad` it is written and stays a
      trainable leaf.
    * ``x`` takes the result's place in the autograd graph, so the gradient
      that reaches whatever produced ``x`` is the op's own derivative.
    * The values land in ``x``'s buffer when a view shares it, so the view's
      base sees them; on Metal, where a view is a value of its own, the
      write is carried back to the base (``lucid._tensor._metal_views``).

    Parameters
    ----------
    x : Tensor
        The tensor written in place and returned.
    name : str
        The op's name, for errors.
    out_of_place : callable
        The op without ``inplace``, given the tensor to read ``x`` from.

    Returns
    -------
    Tensor
        ``x`` itself, now holding the result.

    Raises
    ------
    RuntimeError
        ``x`` is, or is a view of, a leaf that requires grad while autograd
        records; or ``x``'s elements overlap (an expanded view).
    """
    _metal_check(x)
    _write_inplace(x, name, out_of_place)
    _metal_write_back(x)
    return x
