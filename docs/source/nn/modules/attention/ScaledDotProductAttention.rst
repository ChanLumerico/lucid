nn.ScaledDotProductAttention
============================

.. autoclass:: lucid.nn.ScaledDotProductAttention

The `ScaledDotProductAttention` module encapsulates the scaled dot-product attention 
operation commonly used in transformer-based architectures. 
It allows configurable masking, dropout, and causal attention.

Class Signature
---------------

.. code-block:: python

    class lucid.nn.ScaledDotProductAttention(
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: _Scalar | None = None,
    )

Parameters
----------

- **attn_mask** (*Tensor | None*, optional):  
  A mask tensor of shape `(N, H, L, S)`, used to mask out certain positions.  
  If `None`, no masking is applied. Default: `None`.

- **dropout_p** (*float*, optional):  
  Dropout probability applied to attention weights. Default: `0.0`.

- **is_causal** (*bool*, optional):  
  If `True`, applies a causal mask to prevent attending to future positions.  
  This is useful for autoregressive models. Default: `False`.

- **scale** (*_Scalar | None*, optional):  
  Scaling factor applied to the dot-product before softmax.  
  If `None`, the scale is set to `1 / sqrt(D)`, where `D` is the embedding dimension.  
  Default: `None`.

Forward Calculation
-------------------

Given `query`, `key`, and `value` tensors, the module computes attention as follows:

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

**Basic Usage**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> query = Tensor.randn(2, 4, 8, 16)  # Batch=2, Heads=4, Seq_len=8, Dim=16
    >>> key = Tensor.randn(2, 4, 8, 16)
    >>> value = Tensor.randn(2, 4, 8, 16)
    >>> attn = nn.ScaledDotProductAttention()
    >>> output = attn(query, key, value)
    >>> print(output.shape)
    (2, 4, 8, 16)

**Applying a Causal Mask**

.. code-block:: python

    >>> attn = nn.ScaledDotProductAttention(is_causal=True)
    >>> output = attn(query, key, value)
    >>> print(output.shape)
    (2, 4, 8, 16)

.. note::

    - This module is useful for implementing attention layers in transformers.
    - Supports dropout regularization for attention weights.
    - If `is_causal=True`, it ensures autoregressive behavior.
