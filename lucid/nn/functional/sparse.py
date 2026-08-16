"""
nn.functional sparse / embedding operations.
"""

from typing import TYPE_CHECKING

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._unsupported import unsupported_if
from lucid.nn.functional.activations import straight_through

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
    unsupported_if(
        max_norm is not None,
        "embedding",
        "max_norm",
        max_norm,
        detail="Rows are never renormalised.",
    )
    unsupported_if(
        scale_grad_by_freq, "embedding", "scale_grad_by_freq", scale_grad_by_freq
    )
    unsupported_if(
        sparse, "embedding", "sparse", sparse, detail="Gradients are always dense."
    )
    check_embedding_indices(x, weight, "embedding")
    pad = padding_idx if padding_idx is not None else -1
    # Normalised, not passed through.  The engine gather reads the index
    # buffer at a fixed width, so an int8 or bool index of more than a
    # handful of entries is read past its own allocation — SIGBUS, the
    # same shape as the scatter_add defect.  int16/int32/int64 happened to
    # survive; relying on that is relying on an over-read staying inside
    # the page.
    idx = x if x.dtype == lucid.int64 else x.to(lucid.int64)
    return _wrap(_C_engine.nn.embedding(_unwrap(weight), _unwrap(idx), pad))


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


def nearest_codebook(x: Tensor, codebook: Tensor) -> Tensor:
    r"""Index of the closest codebook entry for each row of ``x``.

    .. math::

        k_i = \arg\min_j \big\| x_i - e_j \big\|_2

    The search is the non-differentiable half of vector quantisation:
    the result is an integer field, so no gradient flows through it and
    none is defined.  :func:`vector_quantize` pairs it with the
    straight-through estimator to make the surrounding network trainable.

    Parameters
    ----------
    x : Tensor
        Query field of shape ``(*, D)``.  Only the trailing axis is
        treated as the feature dimension; everything before it is
        flattened into the search's row axis.
    codebook : Tensor
        Codebook of shape ``(K, D)``.

    Returns
    -------
    Tensor
        ``int64`` index field of shape ``(*)`` with values in ``[0, K)``.

    Notes
    -----
    Distances go through :func:`lucid.cdist`, whose ``p=2`` path uses the
    stable expansion :math:`\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2ab^\top`
    rather than materialising an ``(N, K, D)`` difference.  It does
    still build the ``(N, K)`` matrix, which at a large latent grid and
    a large codebook is the dominant allocation of a quantiser; a fused
    engine kernel that reduces over ``K`` without materialising it is the
    natural next step and would slot in behind this exact signature.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn.functional as F
    >>> codebook = lucid.tensor([[0.0, 0.0], [1.0, 1.0]])
    >>> x = lucid.tensor([[0.9, 1.1], [0.1, 0.0]])
    >>> F.nearest_codebook(x, codebook).tolist()
    [1, 0]
    """
    if codebook.ndim != 2:
        raise ValueError(f"codebook must be 2-D (K, D), got shape {codebook.shape}")
    dim = int(codebook.shape[-1])
    if int(x.shape[-1]) != dim:
        raise ValueError(
            f"x's trailing axis must match the codebook's, got "
            f"{int(x.shape[-1])} and {dim}"
        )

    lead = tuple(int(s) for s in x.shape[:-1])
    flat = x.reshape(-1, dim)
    idx = lucid.argmin(lucid.cdist(flat, codebook), dim=1)
    return idx.reshape(*lead) if lead else idx


def vector_quantize(x: Tensor, codebook: Tensor) -> tuple[Tensor, Tensor]:
    r"""Snap ``x`` to its nearest codebook entries, straight-through.

    The functional core of :class:`lucid.nn.VectorQuantizer` — van den
    Oord, Vinyals, and Kavukcuoglu, *"Neural Discrete Representation
    Learning"* (2017).  Each row of ``x`` is replaced by the closest
    entry of ``codebook``, and the result is routed through
    :func:`straight_through` so the producer of ``x`` trains as though
    quantisation were the identity.

    Parameters
    ----------
    x : Tensor
        Field of shape ``(*, D)``; the trailing axis is the feature
        dimension.
    codebook : Tensor
        Codebook of shape ``(K, D)``.

    Returns
    -------
    quantized : Tensor
        Shape ``(*, D)``, numerically equal to the selected entries and
        differentiable with respect to ``x``.
    indices : Tensor
        ``int64`` field of shape ``(*)`` naming the selected entries.

    Notes
    -----
    The returned ``quantized`` carries **no** gradient to ``codebook`` —
    the straight-through path routes past it by construction.  Training
    the codebook needs the separate term
    :math:`\|\mathrm{sg}[x] - e\|_2^2`, which
    :class:`lucid.nn.VectorQuantizer` builds and returns alongside the
    commitment term.  Calling this function directly and optimising only
    a reconstruction loss leaves the codebook frozen at its
    initialisation — a silent failure that looks like a model which
    simply will not learn.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn.functional as F
    >>> codebook = lucid.tensor([[0.0, 0.0], [1.0, 1.0]])
    >>> x = lucid.tensor([[0.9, 1.1]], requires_grad=True)
    >>> quantized, indices = F.vector_quantize(x, codebook)
    >>> quantized.tolist(), indices.tolist()
    ([[1.0, 1.0]], [1])
    """
    indices = nearest_codebook(x, codebook)
    dim = int(codebook.shape[-1])
    lead = tuple(int(s) for s in x.shape[:-1])
    hard = embedding(indices.reshape(-1), codebook).reshape(*lead, dim)
    return straight_through(hard, x), indices


def two_hot(values: Tensor, bins: Tensor) -> Tensor:
    r"""Encode continuous scalars as a distribution over two adjacent bins.

    The generalisation of :func:`one_hot` to values that fall *between*
    classes: all entries are zero except at the two bins bracketing the
    scalar, which carry linearly interpolated weight summing to one.

    Parameters
    ----------
    values : Tensor
        Scalars to encode, any shape ``(...)``.
    bins : Tensor
        Bin locations, ``(K,)``, strictly increasing.  Values outside the
        range are clamped onto the end bins.

    Returns
    -------
    Tensor
        ``(..., K)``, non-negative and summing to 1 along the last axis.

    Notes
    -----
    Why a regression head would want this: predicting a scalar with a
    squared error forces the network to commit to one number, and the
    mean of a bimodal target is a value the target never takes.  A
    distribution over bins can represent "either 0 or 10", and the scalar
    is recovered afterwards as :math:`\sum_k p_k b_k`.  Trained with a
    cross-entropy, the gradient no longer scales with the error either,
    which is what lets one set of hyperparameters cover reward scales
    that differ by orders of magnitude.

    Exact on a bin: a value landing on ``bins[k]`` puts all its weight
    there, so this reduces to :func:`one_hot` on the grid.

    See Also
    --------
    one_hot : The discrete case.
    lucid.nn.functional.symlog : Usually applied before encoding, so that
        the grid can be uniform while the bins are not.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn.functional as F
    >>> F.two_hot(lucid.tensor([1.5]), lucid.tensor([0.0, 1.0, 2.0]))
    tensor([[0., 0.5, 0.5]])
    """
    count = int(bins.shape[0])
    if bins.ndim != 1 or count < 2:
        raise ValueError(
            f"bins must be a 1-D tensor of at least two entries, got shape "
            f"{tuple(int(s) for s in bins.shape)}"
        )

    lower, upper = float(bins[0].item()), float(bins[count - 1].item())
    clamped = values.clip(lower, upper).unsqueeze(dim=-1)

    # Weight every bin by how close it is, then keep only the two that
    # bracket the value: `below` counts the bins at or under it, so
    # `below - 1` and `below` are the neighbours.
    below = (bins <= clamped).to(clamped.dtype).sum(dim=-1) - 1.0
    below = below.clip(0.0, float(count - 2))
    index = below.unsqueeze(dim=-1)

    positions = lucid.arange(0, count, 1, dtype=clamped.dtype, device=clamped.device)
    left = (positions == index).to(clamped.dtype)
    right = (positions == index + 1.0).to(clamped.dtype)

    left_edge = (left * bins).sum(dim=-1, keepdim=True)
    right_edge = (right * bins).sum(dim=-1, keepdim=True)
    span = (right_edge - left_edge).clip(1e-12, None)
    weight = ((clamped - left_edge) / span).clip(0.0, 1.0)

    return left * (1.0 - weight) + right * weight
