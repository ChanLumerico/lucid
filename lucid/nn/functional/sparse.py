"""
nn.functional sparse / embedding operations.
"""

from typing import TYPE_CHECKING

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def check_embedding_indices(x: Tensor, weight: Tensor, op: str) -> None:
    """Raise ``IndexError`` if any index falls outside the embedding table.

    The engine gather does no bounds checking: an out-of-range index reads past
    the table and returns whatever is there — zeros for a small overrun (so a
    wrong ``vocab_size`` or an unclamped token id silently trains on empty
    embeddings) and a SIGSEGV once the offset is large.  One reduction over the
    index tensor is negligible next to the gather it guards.

    Shared by :func:`embedding` and :func:`~lucid.nn.functional.embedding_bag`
    so the two cannot disagree on what a valid index is.

    Parameters
    ----------
    x : Tensor
        Index tensor of any shape.  Empty tensors pass trivially.  Non-int64
        indices are cast before the reduction, since the CPU reduce kernels
        do not cover every integer width.
    weight : Tensor
        Embedding table of shape ``(num_embeddings, embedding_dim)``; only
        its leading dimension is consulted.
    op : str
        Name of the calling op, used as the prefix of the raised message.

    Raises
    ------
    IndexError
        If any index is negative or ``>= weight.shape[0]``.
    """
    # Dtype first, and before the range check rather than after.  A float
    # index tensor passes the range check trivially — its values sit in
    # [0, 1) — and then the engine gather reads those float bits as
    # integers, producing an offset that is nowhere near the table.
    # ``F.embedding(float_tensor, weight)`` took the process down with a
    # segmentation fault, which is how the audit's own sweep died.
    if x.dtype not in (lucid.int8, lucid.int16, lucid.int32, lucid.int64, lucid.bool):
        raise TypeError(
            f"{op}: indices must be an integer tensor, got {x.dtype}; "
            f"cast with .to(lucid.int64) first"
        )
    if x.numel() == 0:
        return
    num_embeddings = int(weight.shape[0])
    # The CPU reduce kernels do not cover every integer width (int32 min/max
    # raises), and index tensors legitimately arrive as int32 — normalise to
    # the canonical index dtype before reducing.
    idx = x if x.dtype == lucid.int64 else x.to(lucid.int64)
    lo = int(idx.min().item())
    hi = int(idx.max().item())
    if lo < 0 or hi >= num_embeddings:
        bad = lo if lo < 0 else hi
        raise IndexError(
            f"{op}: index {bad} is out of range for a table with "
            f"{num_embeddings} entries (valid range [0, {num_embeddings - 1}])"
        )


def embedding(
    x: Tensor,
    weight: Tensor,
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    r"""Look up rows of an embedding table by integer indices.

    A learned embedding table maps integer tokens / categorical features
    into dense vectors:

    .. math::

        \mathrm{out}[i_1, \dots, i_k] = W[\, x[i_1, \dots, i_k]\, ]

    where ``W`` of shape ``(num_embeddings, embedding_dim)`` is the
    lookup table and ``x`` holds integer indices in
    ``[0, num_embeddings)``.  Equivalent to a one-hot matmul
    :math:`\mathrm{onehot}(x) W` but computed with an indexed gather.

    Parameters
    ----------
    x : Tensor
        Integer index tensor of arbitrary shape ``(*)``.
    weight : Tensor
        Embedding table of shape ``(num_embeddings, embedding_dim)``.
    padding_idx : int, optional
        If given, the embedding vector at ``weight[padding_idx]`` is
        treated as a padding slot: its gradient is forced to zero so the
        padding embedding stays at its initialised value (typically a
        zero vector) throughout training.
    max_norm : float, optional
        If given, every entry of ``weight`` whose :math:`L_p` norm
        exceeds ``max_norm`` is renormalised in-place to have norm
        ``max_norm`` prior to the lookup (with :math:`p` = ``norm_type``).
    norm_type : float, optional
        The :math:`p` value of the :math:`L_p` norm used by ``max_norm``.
        Default ``2.0``.
    scale_grad_by_freq : bool, optional
        If ``True``, scale gradients of each embedding row by the inverse
        of its frequency in the mini-batch — useful for highly skewed
        token distributions.
    sparse : bool, optional
        Request a sparse gradient w.r.t. ``weight``.  Lucid currently
        always produces a dense gradient; this flag is accepted for
        API compatibility.

    Returns
    -------
    Tensor
        Embedded tensor of shape ``(*, embedding_dim)``.

    Notes
    -----
    The backward pass for ``embedding`` accumulates gradient contributions
    from repeated indices via scatter-add — multiple tokens of the same
    type in a batch correctly sum into the same row of
    :math:`\partial L / \partial W`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import embedding
    >>> table = lucid.randn(10, 4)              # 10 tokens, dim 4
    >>> ids = lucid.tensor([[1, 2, 4], [4, 3, 2]], dtype=lucid.int64)
    >>> out = embedding(ids, table)
    >>> out.shape
    (2, 3, 4)
    """
    check_embedding_indices(x, weight, "embedding")
    pad = padding_idx if padding_idx is not None else -1
    return _wrap(_C_engine.nn.embedding(_unwrap(weight), _unwrap(x), pad))


def one_hot(tensor: Tensor, num_classes: int = -1) -> Tensor:
    r"""One-hot encode an integer class index tensor.

    Maps each integer entry into a one-hot vector along a new trailing
    axis of size ``num_classes``:

    .. math::

        \mathrm{out}[\ldots, c] =
            \begin{cases} 1 & \text{if } \mathrm{tensor}[\ldots] = c \\
                          0 & \text{otherwise} \end{cases}

    Parameters
    ----------
    tensor : Tensor
        Integer tensor of arbitrary shape ``(*)`` whose entries are class
        indices in ``[0, num_classes)``.
    num_classes : int, optional
        Total number of classes :math:`C`.  If ``-1`` (the default), it
        is inferred as ``tensor.max() + 1``; supplying an explicit value
        avoids a host round-trip and is preferred in hot loops.

    Returns
    -------
    Tensor
        One-hot encoded tensor of shape ``(*, num_classes)`` and integer
        dtype.  Cast to a floating dtype if it will participate in
        gradient-based computation.

    Notes
    -----
    For loss functions like cross-entropy, prefer passing raw integer
    targets to the loss directly — one-hot encoding then immediately
    contracting against a softmax wastes memory and breaks the
    log-sum-exp fused path.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import one_hot
    >>> idx = lucid.tensor([0, 2, 1, 2], dtype=lucid.int64)
    >>> one_hot(idx, num_classes=3)
    Tensor([[1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1]])
    """
    return _wrap(_C_engine.nn.one_hot(_unwrap(tensor), num_classes))
