nn.functional.embedding
=======================

.. autofunction:: lucid.nn.functional.embedding

The `embedding` function converts input indices into a dense representation using an embedding weight matrix.
It is commonly used for handling categorical data, such as word embeddings in NLP models.

Function Signature
------------------

.. code-block:: python

    def embedding(
        input_: Tensor,
        weight: Tensor,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
    ) -> Tensor

Parameters
----------
- **input_** (*Tensor*):
  The input tensor containing indices of shape `(N, ...)`, where each element is an index.

- **weight** (*Tensor*):
  The embedding weight matrix of shape `(num_embeddings, embedding_dim)`.

- **padding_idx** (*int | None*, optional):
  If provided, the embedding at `padding_idx` will be zeroed out. Default is None.

- **max_norm** (*float | None*, optional):
  If provided, each embedding vector is normalized to `max_norm` using `norm_type`. 
  Default is None.

- **norm_type** (*float*, optional):
  The p-norm to use for normalization if `max_norm` is specified. Default is 2.0.

Returns
-------
- **Tensor**:
  The embedded tensor of shape `(N, ..., embedding_dim)`, containing the 
  corresponding embeddings from `weight`.

Embedding Process
-----------------

Given an index tensor `input_`, the function retrieves the corresponding row from `weight`:

.. math::

    \text{output}[i, j, ...] = \text{weight}[\text{input_}[i, j, ...]]

If `padding_idx` is specified, embeddings corresponding to this index are set to zero:

.. math::

    \text{output}[i, j, ...] = 0 \quad \text{if} \quad \text{input_}[i, j, ...] = 
    \text{padding_idx}

If `max_norm` is specified, the output vectors are normalized:

.. math::

    \text{output} = \frac{\text{output} \cdot \text{max_norm}}{||\text{output}||_{p}}

where :math:`||\cdot||_p` represents the p-norm.

Examples
--------

**Basic embedding lookup:**

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_indices = lucid.Tensor([[1, 3, 2], [0, 2, 1]], dtype=int)
    >>> embedding_matrix = lucid.Tensor([
    ...     [0.1, 0.2, 0.3],  # Word 0
    ...     [0.4, 0.5, 0.6],  # Word 1
    ...     [0.7, 0.8, 0.9],  # Word 2
    ...     [1.0, 1.1, 1.2],  # Word 3
    ... ], requires_grad=True)
    >>> output = F.embedding(input_indices, embedding_matrix)
    >>> print(output)
    Tensor([...])

**Using padding index:**

.. code-block:: python

    >>> output_with_padding = F.embedding(input_indices, embedding_matrix, padding_idx=1)
    >>> print(output_with_padding)
    Tensor([...])  # Embeddings at index 1 are zeroed out.

**Using max normalization:**

.. code-block:: python

    >>> output_with_norm = F.embedding(input_indices, embedding_matrix, max_norm=1.0)
    >>> print(output_with_norm)
    Tensor([...])  # Each embedding vector is scaled to have norm ≤ 1.0.

.. note::

    - The function ensures proper differentiation when `weight.requires_grad=True`.
    - `padding_idx` ensures that a given index is zeroed in the output tensor.
    - `max_norm` helps regulate the magnitude of embedding vectors, 
      often improving training stability.

