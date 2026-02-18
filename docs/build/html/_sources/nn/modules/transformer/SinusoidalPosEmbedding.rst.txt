nn.SinusoidalPosEmbedding
=========================

.. autoclass:: lucid.nn.SinusoidalPosEmbedding

Overview
--------
`SinusoidalPosEmbedding` adds deterministic sinusoidal positional encodings to
embedding tensors. It supports both unbatched and batched inputs and can either
infer sequence shape dynamically or enforce fixed shape from constructor arguments.

Class Signature
---------------

.. code-block:: python

    class lucid.nn.SinusoidalPosEmbedding(
        seq_len: int | None = None, embed_dim: int | None = None
    )

Parameters
----------

- **seq_len** (*int | None*, optional):
  Fixed sequence length. If `None`, length is inferred from each input.

- **embed_dim** (*int | None*, optional):
  Fixed embedding dimension. If `None`, embedding size is inferred from each input.

Forward Method
--------------

.. code-block:: python

    def forward(input_: lucid.FloatTensor) -> lucid.FloatTensor

**Input:**

- **input_** (*FloatTensor*):
  Input embedding tensor with shape `(L, D)` or `(N, L, D)`.
  The tensor must have a floating-point dtype.

**Output:**

- **FloatTensor**:
  Tensor with the same shape as input, where sinusoidal position encodings are added.

Behavior
--------

- If `seq_len` or `embed_dim` is `None`, the value is inferred from the input tensor.
- If fixed values are provided and they do not match input shape, `ValueError` is raised.
- If the input dtype is not floating-point, `TypeError` is raised.

Sinusoidal Encoding
-------------------

For position :math:`p` and channel index :math:`i`:

.. math::

    \text{PE}(p, 2i) = \sin\left(\frac{p}{10000^{2i / D}}\right), \quad
    \text{PE}(p, 2i + 1) = \cos\left(\frac{p}{10000^{2i / D}}\right)

The module computes :math:`\text{PE}` and returns:

.. math::

    \mathbf{Y} = \mathbf{X} + \text{PE}

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> pos = nn.SinusoidalPosEmbedding()
    >>> x = lucid.random.randn(2, 8, 64)   # (batch, seq_len, embed_dim)
    >>> y = pos(x)
    >>> print(y.shape)
    (2, 8, 64)

    >>> pos_fixed = nn.SinusoidalPosEmbedding(seq_len=8, embed_dim=64)
    >>> y2 = pos_fixed(x)
    >>> print(y2.shape)
    (2, 8, 64)
