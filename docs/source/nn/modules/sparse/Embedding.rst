nn.Embedding
============

.. autoclass:: lucid.nn.Embedding

The `Embedding` module provides a trainable lookup table that maps indices into dense vectors.
It is commonly used in NLP models for word embeddings and categorical feature representations.

Class Signature
---------------

.. code-block:: python

    class lucid.nn.Embedding(
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        _weight: Tensor | None = None,
    )

Parameters
----------
- **num_embeddings** (*int*):
  The size of the vocabulary (number of unique tokens or categories).

- **embedding_dim** (*int*):
  The size of each embedding vector.

- **padding_idx** (*int | None*, optional):
  If provided, the embedding at this index is zeroed out. Default is None.

- **max_norm** (*float | None*, optional):
  If provided, each embedding vector is normalized to `max_norm` using `norm_type`. 
  Default is None.

- **norm_type** (*float*, optional):
  The p-norm to use for normalization if `max_norm` is specified. Default is 2.0.

- **_weight** (*Tensor | None*, optional):
  If provided, uses this predefined embedding weight matrix instead of random initialization.

Returns
-------
- **Embedding Module**:
  A module that allows embedding lookup operations for input indices.

Embedding Lookup:

.. math::

    \text{output}[i, j, ...] = \text{weight}[\text{input_}[i, j, ...]]

If `padding_idx` is specified, embeddings corresponding to this index are set to zero.

If `max_norm` is specified, each output embedding is normalized:

.. math::

    \text{output} = \frac{\text{output} \cdot \text{max_norm}}{||\text{output}||_{p}}

where :math:`||\cdot||_p` represents the p-norm.

Examples
--------

**Basic embedding lookup:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_indices = lucid.Tensor([[1, 3, 2], [0, 2, 1]], dtype=np.int32)
    >>> embedding_layer = nn.Embedding(num_embeddings=4, embedding_dim=3)
    >>> output = embedding_layer(input_indices)
    >>> print(output)
    Tensor([...])

**Using padding index:**

.. code-block:: python

    >>> embedding_layer = nn.Embedding(num_embeddings=4, embedding_dim=3, padding_idx=1)
    >>> output_with_padding = embedding_layer(input_indices)
    >>> print(output_with_padding)
    Tensor([...])  # Embeddings at index 1 are zeroed out.

**Using max normalization:**

.. code-block:: python

    >>> embedding_layer = nn.Embedding(num_embeddings=4, embedding_dim=3, max_norm=1.0)
    >>> output_with_norm = embedding_layer(input_indices)
    >>> print(output_with_norm)
    Tensor([...])  # Each embedding vector is scaled to have norm ≤ 1.0.

.. note::

  - If `_weight` is provided, it must match the shape `(num_embeddings, embedding_dim)`.
  - The module supports backpropagation and can be optimized as part of a neural network.
  - `padding_idx` ensures that a specific embedding index remains zero, which is useful for padding sequences.
  - `max_norm` helps regulate the magnitude of embedding vectors, improving stability during training.

