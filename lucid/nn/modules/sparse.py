"""
Sparse / embedding modules.
"""

from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import empty
import lucid as _lucid
import lucid.nn.init as init
from lucid.nn.functional.sparse import embedding


class Embedding(Module):
    r"""Learnable dense lookup table that maps integer token indices to vectors.

    An embedding table is a matrix :math:`W \in \mathbb{R}^{V \times D}`
    where :math:`V` = ``num_embeddings`` (vocabulary size) and
    :math:`D` = ``embedding_dim``.  The forward pass is a simple index
    operation:

    .. math::

        y = W[\text{idx}]

    Each row :math:`W[i]` is the dense representation (embedding vector)
    of token :math:`i`.  Because indexing is not differentiable with
    respect to the integer indices themselves, gradients flow only into
    the rows of :math:`W` that were selected during the forward pass.

    **Padding index.** When ``padding_idx`` is set, the corresponding
    row is initialised to zero and its gradient is masked to zero during
    backpropagation.  This is the standard way to mark a special
    ``<PAD>`` token whose embedding should never influence the model.

    **Max-norm renormalisation.** When ``max_norm`` is set, every row
    whose :math:`L_p`-norm exceeds ``max_norm`` is rescaled *in place*
    at each forward call before the lookup:

    .. math::

        W[i] \leftarrow W[i] \cdot \frac{\texttt{max\_norm}}{\|W[i]\|_p}
        \quad \text{if } \|W[i]\|_p > \texttt{max\_norm}

    Parameters
    ----------
    num_embeddings : int
        Size of the embedding dictionary (vocabulary size :math:`V`).
    embedding_dim : int
        Dimensionality of each embedding vector (:math:`D`).
    padding_idx : int or None, optional
        If provided, the embedding for this index is fixed at zero and
        receives no gradient updates.  Negative values are normalised
        to ``padding_idx + num_embeddings``.  Default: ``None``.
    max_norm : float or None, optional
        If provided, rows with :math:`L_p`-norm exceeding this value
        are renormalised in place at every forward call.
        Default: ``None``.
    norm_type : float, optional
        The :math:`p` in the :math:`L_p`-norm used by ``max_norm``.
        Default: ``2.0``.
    scale_grad_by_freq : bool, optional
        Not yet implemented.  Raises :exc:`NotImplementedError` if
        ``True``.  Default: ``False``.
    sparse : bool, optional
        Accepted for API compatibility; sparse gradient emission is
        not yet supported.  Default: ``False``.
    device : DeviceLike, optional
        Device for the weight tensor.
    dtype : DTypeLike, optional
        Data type for the weight tensor.

    Attributes
    ----------
    weight : Parameter, shape ``(num_embeddings, embedding_dim)``
        The embedding matrix :math:`W`.  Rows are initialised from
        :math:`\mathcal{N}(0, 1)`.  If ``padding_idx`` is set, that
        row is zeroed immediately after initialisation.

    Shape
    -----
    * **Input** ``x``: ``(*)`` — integer tensor of arbitrary shape
      with values in ``[0, num_embeddings)``.
    * **Output**: ``(*, embedding_dim)`` — the input shape with an
      extra trailing dimension of size ``embedding_dim``.

    Notes
    -----
    The weight matrix is initialised from a standard normal distribution
    :math:`\mathcal{N}(0, 1)`, which gives each embedding a unit-order
    magnitude.  For downstream layers that are sensitive to input scale
    (e.g. transformers), consider dividing by :math:`\sqrt{D}` after
    construction.

    Embedding is commonly used in natural-language processing (token
    embeddings, position encodings), recommendation systems (item /
    user embeddings), and any domain where discrete categorical inputs
    must be projected into a continuous representation space.

    Examples
    --------
    Simple token embedding for a vocabulary of 100 with 16-dim vectors:

    >>> import lucid, lucid.nn as nn
    >>> emb = nn.Embedding(num_embeddings=100, embedding_dim=16)
    >>> idx = lucid.tensor([[1, 5, 3], [0, 2, 7]], dtype=lucid.int64)
    >>> y = emb(idx)
    >>> y.shape    # (2, 3, 16)
    (2, 3, 16)

    Using ``padding_idx`` to mark a ``<PAD>`` token (index 0):

    >>> emb_pad = nn.Embedding(50, 8, padding_idx=0)
    >>> # Row 0 is always zero and never updated
    >>> import lucid.linalg
    >>> float(lucid.linalg.norm(emb_pad.weight[0])) == 0.0
    True
    >>> idx2 = lucid.tensor([0, 3, 0, 7], dtype=lucid.int64)
    >>> emb_pad(idx2).shape
    (4, 8)

    See Also
    --------
    EmbeddingBag : Efficient embedding lookup with per-bag reduction.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the Embedding module. See the class docstring for parameter semantics."""
        super().__init__()
        if scale_grad_by_freq:
            raise NotImplementedError(
                "Embedding(scale_grad_by_freq=True) is not supported yet. "
                "Apply frequency weighting manually after backward()."
            )
        if padding_idx is not None and not (
            -num_embeddings <= padding_idx < num_embeddings
        ):
            raise ValueError(
                f"padding_idx must be within [-{num_embeddings}, "
                f"{num_embeddings}); got {padding_idx}"
            )
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim
        # Normalise negative padding_idx for downstream comparisons.
        self.padding_idx: int | None = (
            padding_idx + num_embeddings
            if padding_idx is not None and padding_idx < 0
            else padding_idx
        )
        self.max_norm: float | None = max_norm
        self.norm_type: float = norm_type
        self.scale_grad_by_freq: bool = scale_grad_by_freq
        self.sparse: bool = sparse
        self.weight: Parameter = Parameter(
            empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        )
        init.normal_(self.weight)
        # Zero out the pad row on init — matches the reference framework
        # so untouched models do not leak random values through the pad slot.
        if self.padding_idx is not None:
            self._zero_pad_row()

    def _zero_pad_row(self) -> None:
        """Set ``weight[padding_idx]`` to zero in-place via engine ops.

        Cheap because it only fires from ``__init__``; runtime forward
        does not need this.
        """
        # ``index_fill`` on dim=0 zeroes the chosen row; rebind ``weight._impl``
        # so requires_grad / parameter identity are preserved.
        new_w: Tensor = _lucid.index_fill(
            self.weight,
            0,  # type: ignore[arg-type]
            _lucid.tensor(
                [int(self.padding_idx)], dtype=_lucid.int64, device=self.weight.device  # type: ignore[arg-type]
            ),
            0.0,  # type: ignore[arg-type]
        )
        self.weight._impl = new_w._impl

    def _renorm_weight_inplace(self) -> None:
        """Apply ``max_norm`` rescaling to rows that exceed the cap."""
        w: Tensor = self.weight
        # Per-row Lp-norm via engine ops.
        if self.norm_type == 2.0:
            norms: Tensor = (w * w).sum(dim=1).sqrt()
        elif self.norm_type == 1.0:
            norms = w.abs().sum(dim=1)
        else:
            norms = (w.abs() ** float(self.norm_type)).sum(dim=1) ** (
                1.0 / float(self.norm_type)
            )
        scale_raw: Tensor = float(self.max_norm) / (norms + 1e-7)  # type: ignore[arg-type]
        ones: Tensor = _lucid.ones_like(scale_raw)
        scale: Tensor = scale_raw.minimum(ones).unsqueeze(-1)
        new_w: Tensor = w * scale
        self.weight._impl = new_w._impl

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        """Look up embeddings for the given indices.

        Parameters
        ----------
        input : Tensor
            Tensor of integer indices.

        Returns
        -------
        Tensor
            Tensor of embedding vectors of shape ``(*input.shape, embedding_dim)``.
        """
        if self.max_norm is not None:
            self._renorm_weight_inplace()
        return embedding(x, self.weight, self.padding_idx)

    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        s: str = f"{self.num_embeddings}, {self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        if self.max_norm is not None:
            s += f", max_norm={self.max_norm}"
        if self.norm_type != 2.0:
            s += f", norm_type={self.norm_type}"
        return s


class EmbeddingBag(Module):
    r"""Embedding lookup table with per-bag reduction.

    Computes a summary embedding for each *bag* (a variable-length set
    of token indices) without materialising the full per-token embedding
    tensor.  Given an embedding matrix
    :math:`W \in \mathbb{R}^{V \times D}` and a bag of indices
    :math:`\mathcal{B} = \{i_1, i_2, \ldots, i_k\}`, the output
    embedding for that bag is:

    .. math::

        y =
        \begin{cases}
            \displaystyle\sum_{j \in \mathcal{B}} W[j]
              & \text{mode} = \texttt{'sum'} \\[6pt]
            \displaystyle\frac{1}{|\mathcal{B}|}\sum_{j \in \mathcal{B}} W[j]
              & \text{mode} = \texttt{'mean'} \\[6pt]
            \displaystyle\max_{j \in \mathcal{B}} W[j]
              & \text{mode} = \texttt{'max'}
        \end{cases}

    where the :math:`\max` is taken element-wise across the embedding
    dimension.

    **2-D input (fixed-length bags).** When ``offsets`` is ``None`` the
    input ``x`` must have shape ``(B, L)`` — ``B`` bags each containing
    exactly ``L`` indices.  The reduction is applied over the ``L``
    axis, yielding an output of shape ``(B, D)``.

    **1-D input with offsets (variable-length bags).** When ``offsets``
    is provided, ``x`` is a flat 1-D integer tensor of all indices
    concatenated, and ``offsets`` is a 1-D integer tensor of length
    ``B`` marking the start position of each bag.  Bag :math:`b`
    consists of indices ``x[offsets[b] : offsets[b+1]]`` (the last bag
    runs to the end of ``x``).

    Parameters
    ----------
    num_embeddings : int
        Size of the embedding dictionary (vocabulary size :math:`V`).
    embedding_dim : int
        Dimensionality of each embedding vector (:math:`D`).
    max_norm : float or None, optional
        If provided, rows with :math:`L_p`-norm exceeding this value
        are renormalised before the lookup (see :class:`Embedding`).
        Default: ``None``.
    norm_type : float, optional
        The :math:`p` for ``max_norm`` renormalisation.  Default: ``2.0``.
    scale_grad_by_freq : bool, optional
        Accepted for API compatibility; not yet implemented.
        Default: ``False``.
    mode : {'sum', 'mean', 'max'}, optional
        Reduction applied over each bag.  Default: ``'mean'``.
    sparse : bool, optional
        Accepted for API compatibility; sparse gradient emission is
        not yet supported.  Default: ``False``.
    padding_idx : int or None, optional
        Indices equal to ``padding_idx`` contribute zero to the
        reduction and do not receive gradient updates.
        Default: ``None``.
    device : DeviceLike, optional
        Device for the weight tensor.
    dtype : DTypeLike, optional
        Data type for the weight tensor.

    Attributes
    ----------
    weight : Parameter, shape ``(num_embeddings, embedding_dim)``
        The embedding matrix :math:`W`, initialised from
        :math:`\mathcal{N}(0, 1)`.

    Shape
    -----
    * **x** (2-D path): ``(B, L)`` integer indices → output ``(B, D)``.
    * **x** (1-D + offsets path): ``(total_indices,)`` integer indices,
      ``offsets`` ``(B,)`` → output ``(B, D)``.

    Notes
    -----
    :class:`EmbeddingBag` is more memory-efficient than computing
    ``Embedding(x).sum(dim=1)`` because it fuses the lookup and
    reduction into a single kernel call, avoiding the intermediate
    ``(B, L, D)`` tensor.  This is especially beneficial for large
    vocabularies and long bags.

    ``'max'`` mode is not differentiable with respect to ties; the
    gradient is propagated only through the index that achieved the
    maximum.

    Examples
    --------
    Fixed-length bags (2-D input), mean pooling:

    >>> import lucid, lucid.nn as nn
    >>> emb_bag = nn.EmbeddingBag(num_embeddings=20, embedding_dim=8, mode='mean')
    >>> idx = lucid.tensor([[1, 3, 5], [0, 2, 4]], dtype=lucid.int64)  # (B=2, L=3)
    >>> y = emb_bag(idx)
    >>> y.shape    # (B=2, D=8)
    (2, 8)

    Variable-length bags via offsets (sum pooling):

    >>> emb_sum = nn.EmbeddingBag(10, 4, mode='sum')
    >>> flat_idx = lucid.tensor([0, 1, 2, 3, 4, 5], dtype=lucid.int64)  # 6 total
    >>> offsets = lucid.tensor([0, 2, 5], dtype=lucid.int64)  # bags: [0,1], [2,3,4], [5]
    >>> y2 = emb_sum(flat_idx, offsets)
    >>> y2.shape    # (B=3, D=4)
    (3, 4)

    See Also
    --------
    Embedding : Per-token embedding lookup without reduction.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        padding_idx: int | None = None,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the EmbeddingBag module. See the class docstring for parameter semantics."""
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.padding_idx = padding_idx
        self.weight = Parameter(
            empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        )
        init.normal_(self.weight)

    def forward(self, x: Tensor, offsets: Tensor | None = None) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        """Look up embeddings for the given indices.

        Parameters
        ----------
        input : Tensor
            Tensor of integer indices.

        Returns
        -------
        Tensor
            Tensor of embedding vectors of shape ``(*input.shape, embedding_dim)``.
        """
        from lucid.nn.functional.sampling import embedding_bag as _eb

        _mode_map = {"sum": "sum", "mean": "mean", "max": "max"}
        return _eb(
            x,
            self.weight,
            offsets=offsets,
            mode=_mode_map.get(self.mode, "mean"),
            padding_idx=self.padding_idx,
        )

    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"{self.num_embeddings}, {self.embedding_dim}, "
            f"mode={self.mode!r}, padding_idx={self.padding_idx}"
        )
