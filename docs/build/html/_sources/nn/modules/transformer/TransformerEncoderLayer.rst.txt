nn.TransformerEncoderLayer
==========================

.. autoclass:: lucid.nn.TransformerEncoderLayer

Overview
--------
The `TransformerEncoderLayer` module implements a single layer of the Transformer encoder,
which consists of multi-head self-attention followed by a feedforward network.
Both sublayers include residual connections and layer normalization.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.TransformerEncoderLayer(
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        bias: bool = True,
    )

Parameters
----------
- **d_model** (*int*):
  The dimensionality of the input embeddings (:math:`d_{model}`).

- **num_heads** (*int*):
  The number of attention heads (:math:`H`).

  .. warning::

     The embedding dimension (:math:`d_{model}`) must be divisible by :math:`H`.

- **dim_feedforward** (*int*, optional, default=2048):
  The dimensionality of the intermediate layer in the feedforward network.

- **dropout** (*float*, optional, default=0.1):
  Dropout probability applied to the attention and feedforward layers.

- **activation** (*Callable[[Tensor], Tensor]*, optional, default=F.relu):
  The activation function applied in the feedforward network.

- **layer_norm_eps** (*float*, optional, default=1e-5):
  A small constant added to the denominator for numerical stability in layer normalization.

- **norm_first** (*bool*, optional, default=False):
  If `True`, applies layer normalization before the attention and feedforward sublayers, 
  instead of after.

- **bias** (*bool*, optional, default=True):
  If `True`, enables bias terms in the linear layers.

Forward Method
--------------

.. code-block:: python

    def forward(
        src: Tensor, 
        src_mask: Tensor | None = None, 
        src_key_padding_mask: Tensor | None = None, 
        is_causal: bool = False
    ) -> Tensor

Computes the forward pass of the Transformer encoder layer.

**Inputs:**

- **src** (*Tensor*):
  The input tensor of shape :math:`(N, L, d_{model})`, 
  where:
  - :math:`N` is the batch size.
  - :math:`L` is the sequence length.
  - :math:`d_{model}` is the embedding dimension.

- **src_mask** (*Tensor | None*, optional):
  A mask of shape :math:`(L, L)` applied to attention weights to prevent 
  attending to certain positions. Default is `None`.

- **src_key_padding_mask** (*Tensor | None*, optional):
  A mask of shape :math:`(N, L)`, where non-zero values indicate positions 
  that should be ignored. Default is `None`.

- **is_causal** (*bool*, optional, default=False):
  If `True`, enforces a lower-triangular mask to prevent positions from 
  attending to future positions.

**Output:**

- **Tensor**: The output tensor of shape :math:`(N, L, d_{model})`.

Mathematical Details
--------------------
The Transformer encoder layer consists of the following computations:

1. **Multi-Head Self-Attention**
   
   The input undergoes multi-head self-attention:
   
   .. math::

       A = \operatorname{softmax} \left( \frac{QK^T}{\sqrt{d_h}} + M \right) V
   
   where:
   - :math:`Q, K, V` are the query, key, and value matrices.
   - :math:`M` is the optional attention mask.
   - :math:`d_h = \frac{d_{model}}{H}` is the per-head embedding size.

2. **Feedforward Network**
   
   The attention output passes through a two-layer feedforward network with an 
   activation function:
   
   .. math::

       F(x) = \operatorname{Activation}(x W_1 + b_1) W_2 + b_2
   
   where :math:`W_1, W_2, b_1, b_2` are learnable parameters.

3. **Layer Normalization and Residual Connections**
   
   - If `norm_first=False`:
     
     .. math::

         y = \operatorname{LayerNorm}(x + \operatorname{SelfAttention}(x))
         z = \operatorname{LayerNorm}(y + \operatorname{FeedForward}(y))
   
   - If `norm_first=True`:
     
     .. math::

         y = x + \operatorname{SelfAttention}(\operatorname{LayerNorm}(x))
         z = y + \operatorname{FeedForward}(\operatorname{LayerNorm}(y))

Usage Example
-------------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    # Initialize TransformerEncoderLayer with embedding dimension 512 and 8 heads
    encoder_layer = nn.TransformerEncoderLayer(d_model=512, num_heads=8)

    # Create random input tensor
    src = lucid.random.randn(16, 10, 512)  # (batch, seq_len, embed_dim)
    
    # Compute encoder output
    output = encoder_layer(src)
    print(output.shape)  # Expected output: (16, 10, 512)
