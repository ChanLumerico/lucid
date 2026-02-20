nn.RotaryPosEmbedding
=====================

.. autoclass:: lucid.nn.RotaryPosEmbedding

Overview
--------
`RotaryPosEmbedding` applies Rotary Position Embedding (RoPE) to the input tensor
along the last embedding dimension. Unlike the stateless functional API, this
module manages internal cosine/sine caches and reuses them across forward calls.

Class Signature
---------------

.. code-block:: python

    class lucid.nn.RotaryPosEmbedding(
        embed_dim: int | None = None,
        max_seq_len: int | None = None,
        interleaved: bool = True,
    )

Parameters
----------

- **embed_dim** (*int | None*, optional):
  Fixed embedding dimension. If `None`, embedding size is inferred from each input.

- **max_seq_len** (*int | None*, optional):
  Optional cache capacity. If provided, required positions must satisfy
  `required_len <= max_seq_len`.

- **interleaved** (*bool*, optional):
  Rotation layout selector.
  If `True`, pairs adjacent channels `(0,1), (2,3), ...`.
  If `False`, pairs split-halves `(0,D/2), (1,D/2+1), ...`.

Forward Method
--------------

.. code-block:: python

    def forward(
        input_: lucid.FloatTensor,
        position_ids: lucid.LongTensor | None = None,
    ) -> lucid.FloatTensor

**Input:**

- **input_** (*FloatTensor*):
  Input tensor of shape `(..., L, D)`.
  The embedding dimension `D` must be even.

- **position_ids** (*LongTensor | None*, optional):
  Optional explicit positions of shape `(L,)`.
  If `None`, positions are generated as `[0, 1, ..., L-1]`.

**Output:**

- **FloatTensor**:
  Tensor with the same shape as `input_` after RoPE rotation.

RoPE Matrix in 2D
-----------------

For one channel pair and angle :math:`\phi`, RoPE uses:

.. math::

    \mathbf{R}(\phi)=
    \begin{bmatrix}
    \cos\phi & -\sin\phi \\
    \sin\phi & \cos\phi
    \end{bmatrix}

Generalized :math:`D`-Dimensional Rotation
------------------------------------------

Let :math:`D` be even and :math:`\Theta=\{\theta_i\}_{i=1}^{D/2}` with:

.. math::

    \theta_i = 10000^{-2(i-1)/D}, \quad
    \phi_{m,i} = m\theta_i

RoPE at position :math:`m` is the block-diagonal matrix:

.. math::

    \mathbf{R}_{\Theta,m}^{D}
    =
    \bigoplus_{i=1}^{D/2}\mathbf{R}(\phi_{m,i})
    =
    \mathrm{diag}\!\left(
    \mathbf{R}(\phi_{m,1}), \dots, \mathbf{R}(\phi_{m,D/2})
    \right)

Attention Application
---------------------

RoPE is applied after linear projections:

.. math::

    \mathbf{q}_m = \mathbf{R}_{\Theta,m}^{D}\mathbf{W}_q\mathbf{x}_m,\quad
    \mathbf{k}_n = \mathbf{R}_{\Theta,n}^{D}\mathbf{W}_k\mathbf{x}_n

Then:

.. math::

    \mathbf{q}_m^\top \mathbf{k}_n
    =
    \mathbf{x}_m^\top \mathbf{W}_q^\top
    \left(\mathbf{R}_{\Theta,m}^{D}\right)^\top
    \mathbf{R}_{\Theta,n}^{D}
    \mathbf{W}_k\mathbf{x}_n
    =
    \mathbf{x}_m^\top \mathbf{W}_q^\top
    \mathbf{R}_{\Theta,n-m}^{D}
    \mathbf{W}_k\mathbf{x}_n

Thus relative offset :math:`(n-m)` appears directly in attention scores.

Efficient Computational Form
----------------------------

Instead of explicit block matrix multiplication, implementation uses:

.. math::

    \mathrm{RoPE}(\mathbf{x})
    =
    \mathbf{x}\odot\cos\boldsymbol{\phi}
    +
    \mathrm{rotate\_half}(\mathbf{x})\odot\sin\boldsymbol{\phi}

where :math:`\mathrm{rotate\_half}` performs the pairwise signed swap.

Caching Behavior
----------------

This module caches :math:`\cos\boldsymbol{\phi}` and :math:`\sin\boldsymbol{\phi}`
internally as buffers:

- Reuses cached values when sequence positions are already covered.
- Rebuilds cache when required length grows, device changes, or embedding dimension changes.
- If `max_seq_len` is set, cache is built at that capacity and validated on each call.

Validation Rules
----------------

- Raises `TypeError` for non-floating inputs.
- Raises `ValueError` when `D` is odd.
- Raises `ValueError` when fixed `embed_dim` mismatches input.
- Raises `ValueError` when `position_ids` shape is invalid or contains negative values.
- Raises `ValueError` when required positions exceed `max_seq_len`.

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> rope = nn.RotaryPosEmbedding(embed_dim=64, max_seq_len=2048)
    >>> x = lucid.random.randn(2, 16, 64)  # (batch, seq_len, dim)
    >>> y = rope(x)
    >>> print(y.shape)
    (2, 16, 64)

    >>> pos = lucid.arange(16, dtype=lucid.Long)
    >>> y2 = rope(x, position_ids=pos)
    >>> print(y2.shape)
    (2, 16, 64)

