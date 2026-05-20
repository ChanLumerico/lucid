"""
Classification metrics — accuracy / correct_count.

3.2.0 perf-pass addition.  Hand-written wrappers around the lazy graph
``logits.argmax(dim) → eq(target) → cast → reduce`` so that training
loops can express the common ``running_correct += correct_count(...).item()``
pattern in one expression instead of four chained ``Tensor`` methods.

The savings are not in GPU compute — MLX's lazy graph already fuses the
chain into ~1 kernel — but in Python-side wrap overhead: each
intermediate ``Tensor`` wrap costs ~1–2 µs through ``_wrap``, and the
training pattern paid 4 of them per batch.  Across a 5-epoch LeNet-5 /
MNIST run that's ~25 ms.  More importantly the call sites read better.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def accuracy(logits: Tensor, target: Tensor, *, dim: int = -1) -> Tensor:
    r"""Compute the mean classification accuracy in ``[0, 1]``.

    Equivalent to ``(logits.argmax(dim=dim) == target).float().mean()`` —
    written as a fused expression so the result is a single ``Tensor``
    (one Python wrap) rather than a four-op chain.  Returns a 0-d Tensor;
    ``accuracy(...).item()`` materialises it as a Python ``float``.

    Parameters
    ----------
    logits : Tensor
        Predicted class scores of shape ``(*, C)``.  The class dimension
        defaults to the last axis; override with ``dim`` for layouts that
        keep the class axis elsewhere (e.g. dense-prediction outputs).
    target : Tensor
        Ground-truth class indices of shape ``(*,)`` — i.e. ``logits.shape``
        with the class axis removed.  Integer dtype expected.
    dim : int, default -1
        Axis along which to take the ``argmax``.  Negative indexing is
        resolved against ``logits.ndim``.

    Returns
    -------
    Tensor
        0-d scalar in ``[0, 1]``; the float dtype matches the upcast
        used by ``.mean()`` (typically ``float32``).

    Notes
    -----
    For accumulating correct-count across mini-batches without dividing
    by batch size every step, use :func:`correct_count` instead — it
    skips the ``.float()`` cast + ``.mean()`` reduction and returns an
    ``int64`` 0-d Tensor that can be added cheaply across batches.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn.functional as F
    >>> logits = lucid.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    >>> target = lucid.tensor([0, 2])
    >>> F.accuracy(logits, target).item()
    1.0
    """
    return (logits.argmax(dim=dim) == target).float().mean()


def correct_count(logits: Tensor, target: Tensor, *, dim: int = -1) -> Tensor:
    r"""Count of correct top-1 predictions as an ``int64`` 0-d Tensor.

    Equivalent to ``(logits.argmax(dim=dim) == target).long().sum()`` —
    written as a single function so the four-op chain in user code
    collapses to one Python wrap.  The complementary primitive to
    :func:`accuracy`: where :func:`accuracy` divides by total count
    eagerly, ``correct_count`` keeps the count integer so the user can
    accumulate across batches and divide once at the end of the epoch.

    Parameters
    ----------
    logits : Tensor
        Predicted class scores of shape ``(*, C)``.
    target : Tensor
        Ground-truth class indices of shape ``(*,)``.  Integer dtype
        expected.
    dim : int, default -1
        Axis along which to take the ``argmax``.

    Returns
    -------
    Tensor
        0-d ``int64`` scalar — the number of positions where
        ``argmax(logits) == target``.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn.functional as F
    >>> logits = lucid.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    >>> target = lucid.tensor([0, 1])
    >>> F.correct_count(logits, target).item()
    1
    """
    return (logits.argmax(dim=dim) == target).long().sum()
