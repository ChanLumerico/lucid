nn.TransformerEncoder
=====================

.. autoclass:: lucid.nn.TransformerEncoder

Overview
--------
The `TransformerEncoder` module stacks multiple `TransformerEncoderLayer` 
instances to form a complete Transformer encoder. It sequentially processes the 
input data through multiple encoder layers and optionally applies layer normalization.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.TransformerEncoder(
        encoder_layer: TransformerEncoderLayer | nn.Module,
        num_layers: int,
        norm: nn.Module | None = None,
    )

Parameters
----------
- **encoder_layer** (*TransformerEncoderLayer | nn.Module*):
  A single instance of `TransformerEncoderLayer` that will be replicated for 
  `num_layers` times.

- **num_layers** (*int*):
  The number of encoder layers in the stack.

- **norm** (*nn.Module | None*, optional):
  An optional layer normalization module applied to the final output. 
  Default is `None`.

Forward Method
--------------

.. code-block:: python

    def forward(
        src: Tensor, 
        src_mask: Tensor | None = None, 
        src_key_padding: Tensor | None = None, 
        is_causal: bool = False
    ) -> Tensor

Computes the forward pass of the Transformer encoder.

**Inputs:**

- **src** (*Tensor*):
  The input tensor of shape :math:`(N, L, d_{model})`, where:
  - :math:`N` is the batch size.
  - :math:`L` is the sequence length.
  - :math:`d_{model}` is the embedding dimension.

- **src_mask** (*Tensor | None*, optional):
  A mask of shape :math:`(L, L)` applied to attention weights. Default is `None`.

- **src_key_padding** (*Tensor | None*, optional):
  A mask of shape :math:`(N, L)`, where non-zero values indicate positions 
  that should be ignored. Default is `None`.

- **is_causal** (*bool*, optional, default=False):
  If `True`, enforces a lower-triangular mask to prevent positions from 
  attending to future positions.

**Output:**

- **Tensor**: The output tensor of shape :math:`(N, L, d_{model})`.

Mathematical Details
--------------------
The Transformer encoder processes input through a sequence of encoder layers as follows:

1. **Iterative Encoding**
   
   Each input tensor :math:`X` is passed through `num_layers` encoder layers:
   
   .. math::

       X_0 = X
       X_{i+1} = \operatorname{EncoderLayer}(X_i), \quad \forall i 
       \in [0, \text{num\_layers}-1]

2. **Optional Normalization**
   
   If `norm` is provided, it is applied to the final output:
   
   .. math::
    
       Y = \operatorname{LayerNorm}(X_{\text{num\_layers}})
   
   Otherwise, the final encoder layer output is returned.

Usage Example
-------------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    # Create an encoder layer
    encoder_layer = nn.TransformerEncoderLayer(d_model=512, num_heads=8)
    
    # Stack multiple encoder layers into a Transformer encoder
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    # Create random input tensor
    src = lucid.random.randn(16, 10, 512)  # (batch, seq_len, embed_dim)
    
    # Compute encoder output
    output = transformer_encoder(src)
    print(output.shape)  # Expected output: (16, 10, 512)

