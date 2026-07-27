"""Packing a group of differently-shaped tensors into one flat state.

An ODE solver integrates a single tensor, but two things want to integrate a
*collection*: the adjoint method, whose augmented state is the solution
alongside its adjoint and one accumulator per parameter, and a caller whose
state is naturally a tuple.  Both reduce to the same trick — concatenate into
one vector, integrate that, split it back.
"""

import math
from typing import Sequence

import lucid
from lucid._tensor.tensor import Tensor


__all__: list[str] = []


def flatten(tensors: Sequence[Tensor]) -> Tensor:
    """Concatenate a group of tensors into one 1-D vector.

    Parameters
    ----------
    tensors : sequence of Tensor
        Non-empty group, all on the same device.  Shapes may differ.

    Returns
    -------
    Tensor
        A 1-D tensor holding every element in order.

    Notes
    -----
    Differentiable: the concatenation and the reshapes are ordinary ops, so a
    gradient reaching the flat vector routes back to each source tensor.
    """
    return lucid.concat([t.reshape(-1) for t in tensors])


def shapes_of(tensors: Sequence[Tensor]) -> list[tuple[int, ...]]:
    """Record the shapes needed to undo :func:`flatten`.

    Parameters
    ----------
    tensors : sequence of Tensor
        The group about to be flattened.

    Returns
    -------
    list of tuple of int
        One shape per tensor, in order.  Hand this to :func:`unflatten` to
        split a flat state back into its parts.
    """
    return [tuple(t.shape) for t in tensors]


def unflatten(flat: Tensor, shapes: Sequence[tuple[int, ...]]) -> list[Tensor]:
    """Split a flat vector back into tensors of the recorded shapes.

    Parameters
    ----------
    flat : Tensor
        1-D tensor produced by :func:`flatten`.
    shapes : sequence of tuple of int
        The shapes to rebuild, in the order they were flattened.

    Returns
    -------
    list of Tensor
        One tensor per entry of ``shapes``.

    Raises
    ------
    ValueError
        If ``flat`` does not hold exactly as many elements as ``shapes``
        describe.

    Notes
    -----
    Also differentiable, so the pair round-trips inside an autograd graph.
    """
    total = sum(math.prod(shape) for shape in shapes)
    if flat.shape != (total,):
        raise ValueError(
            f"flat state has shape {tuple(flat.shape)} but the recorded shapes "
            f"describe {total} elements"
        )

    pieces: list[Tensor] = []
    offset = 0
    for shape in shapes:
        size = math.prod(shape)
        pieces.append(flat[offset : offset + size].reshape(*shape))
        offset += size
    return pieces
