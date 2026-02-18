nn.functional.sinusoidal_pos_embedding
======================================

.. autofunction:: lucid.nn.functional.sinusoidal_pos_embedding

The `sinusoidal_pos_embedding` function creates deterministic positional encodings
for sequence models. The output can be added directly to token embeddings to inject
position information without learnable position parameters.

Function Signature
------------------

.. code-block:: python

    def sinusoidal_pos_embedding(
        seq_len: int,
        embed_dim: int,
        device: _DeviceType = "cpu",
        dtype: Numeric | None = None,
    ) -> Tensor

Parameters
----------

- **seq_len** (*int*):
  Sequence length :math:`L`. Must be positive.

- **embed_dim** (*int*):
  Embedding dimension :math:`D`. Must be positive.

- **device** (*_DeviceType*, optional):
  Device for the output tensor. Default is `"cpu"`.

- **dtype** (*Numeric | None*, optional):
  Output dtype. If `None`, the function uses `lucid.Float32`.

Returns
-------

- **Tensor**:
  Positional embedding tensor of shape `(seq_len, embed_dim)`.

Sinusoidal Formulation
----------------------

For position :math:`p \in [0, L-1]` and frequency index :math:`i`:

.. math::

    \text{PE}(p, 2i) = \sin\left(\frac{p}{10000^{2i / D}}\right)

.. math::

    \text{PE}(p, 2i + 1) = \cos\left(\frac{p}{10000^{2i / D}}\right)

This matches the implementation using:

.. math::

    \text{div_term}_i = \exp\left(-\log(10000)\frac{2i}{D}\right)
    \quad\Rightarrow\quad
    \theta_{p,i} = p \cdot \text{div_term}_i

and then applying sine to even channels and cosine to odd channels.

Implementation Notes
--------------------

- Raises `ValueError` when `seq_len <= 0` or `embed_dim <= 0`.
- For odd `embed_dim`, the last even channel is filled with `sin` and has no paired `cos` channel.
- The function is deterministic and contains no trainable parameters.

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn.functional as F
    >>> pe = F.sinusoidal_pos_embedding(seq_len=4, embed_dim=8)
    >>> print(pe.shape)
    (4, 8)

    >>> token_embed = lucid.random.randn(2, 4, 8)  # (batch, seq, dim)
    >>> x = token_embed + pe.unsqueeze(axis=0)
    >>> print(x.shape)
    (2, 4, 8)
