nn.functional.scaled_dot_product_attention
==========================================

.. autofunction:: lucid.nn.functional.scaled_dot_product_attention

The `scaled_dot_product_attention` function computes scaled dot-product attention, 
a fundamental operation in transformer-based models.

Function Signature
------------------

.. code-block:: python

    def scaled_dot_product_attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: _Scalar | None = None,
    ) -> Tensor

Parameters
----------

- **query** (*Tensor*):  
  The query tensor of shape `(N, H, L, D)`, where:
  - `N`: Batch size  
  - `H`: Number of attention heads  
  - `L`: Sequence length  
  - `D`: Embedding dimension per head  

- **key** (*Tensor*):  
  The key tensor of shape `(N, H, S, D)`, where `S` is the source sequence length.

- **value** (*Tensor*):  
  The value tensor of shape `(N, H, S, D)`, matching the key tensor.

- **attn_mask** (*Tensor | None*, optional):  
  A mask tensor of shape `(N, H, L, S)`, used to mask out certain positions. Default: `None`.

- **dropout_p** (*float*, optional):  
  Dropout probability applied to attention weights. Default: `0.0`.

- **is_causal** (*bool*, optional):  
  If `True`, applies a causal mask to prevent attending to future positions. Default: `False`.

- **scale** (*_Scalar | None*, optional):  
  Scaling factor applied to the dot-product before the softmax operation.  
  If `None`, the scale is set to `1 / sqrt(D)`. Default: `None`.

Returns
-------

- **Tensor**:  
  The output tensor of shape `(N, H, L, D)`, containing the weighted sum of values.

Attention Mechanism
-------------------

The function performs the following operations:

1. Compute the scaled dot-product scores:

   .. math::

       \text{Scores} = \frac{\mathbf{Q} \mathbf{K}^\top}{\text{scale}}

2. Apply the attention mask if provided:

   .. math::

       \text{Scores} = \text{Scores} + \text{attn_mask}

3. Compute the attention weights using softmax:

   .. math::

       \text{Attn Weights} = \text{softmax}(\text{Scores})

4. Apply dropout (if enabled):

   .. math::

       \text{Attn Weights} = \text{Dropout}(\text{Attn Weights})

5. Compute the output:

   .. math::

       \text{Output} = \text{Attn Weights} \cdot \mathbf{V}

Examples
--------

**Basic Attention Computation**

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> query = lucid.random.randn(2, 4, 8, 16)  # Batch=2, Heads=4, Seq_len=8, Dim=16
    >>> key = lucid.random.randn(2, 4, 8, 16)
    >>> value = lucid.random.randn(2, 4, 8, 16)
    ...
    >>> output = F.scaled_dot_product_attention(query, key, value)
    >>> print(output.shape)
    (2, 4, 8, 16)

.. note::

    - The function supports multi-head attention computations.
    - The optional causal mask ensures autoregressive behavior in transformers.
    - The dropout probability is applied to attention weights before computing the final output.
