nn.LearnedPosEmbedding
======================

.. autoclass:: lucid.nn.LearnedPosEmbedding

Overview
--------
`LearnedPosEmbedding` adds trainable positional embeddings to input embeddings.
It uses an internal `nn.Embedding(max_len, embed_dim)` table indexed by token
positions and supports both unbatched and batched inputs.

Class Signature
---------------

.. code-block:: python

    class lucid.nn.LearnedPosEmbedding(
        max_len: int,
        embed_dim: int,
    )

Parameters
----------

- **max_len** (*int*):
  Maximum supported sequence length.

- **embed_dim** (*int*):
  Embedding dimension of both input and positional embedding.

Forward Method
--------------

.. code-block:: python

    def forward(
        x: lucid.FloatTensor,
        offset: int = 0,
    ) -> lucid.FloatTensor

**Input:**

- **x** (*FloatTensor*):
  Input embedding tensor with shape `(L, D)` or `(N, L, D)`.
  The tensor must have a floating-point dtype.

- **offset** (*int*, optional):
  Start position index. Default is `0`.
  The valid range must satisfy `0 <= offset` and `offset + L <= max_len`.

**Output:**

- **FloatTensor**:
  Tensor with the same shape as `x`, with learned positional embeddings added.

Computation
-----------

For sequence length :math:`L`, the module first creates position indices:

.. math::

    \text{pos_ids} = [\text{offset}, \text{offset}+1, \dots, \text{offset}+L-1]

Then it looks up the positional vectors from the learnable table
:math:`\mathbf{W}_{pos} \in \mathbb{R}^{\text{max_len} \times D}`:

.. math::

    \mathbf{P}_t = \mathbf{W}_{pos}[\text{pos_ids}_t]

Finally it returns:

.. math::

    \mathbf{Y} = \mathbf{X} + \mathbf{P}

Shape and Type Checks
---------------------

- Raises `ValueError` when input is not 2D/3D.
- Raises `TypeError` when input dtype is not floating-point.
- Raises `ValueError` when input `embed_dim` does not match module `embed_dim`.
- Raises `ValueError` when position range from `offset` exceeds `max_len`.

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> pos = nn.LearnedPosEmbedding(max_len=512, embed_dim=64)
    >>> x = lucid.random.randn(2, 16, 64)
    >>> y = pos(x)
    >>> print(y.shape)
    (2, 16, 64)

    >>> y2 = pos(x, offset=32)
    >>> print(y2.shape)
    (2, 16, 64)
