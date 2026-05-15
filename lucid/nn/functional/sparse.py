"""
nn.functional sparse / embedding operations.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


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
