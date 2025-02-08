nn.TransformerDecoderLayer
==========================

.. autoclass:: lucid.nn.TransformerDecoderLayer

Overview
--------
The `TransformerDecoderLayer` module implements a single layer of the Transformer decoder,
which consists of masked multi-head self-attention, multi-head cross-attention, and a feedforward network.
Each sublayer includes residual connections and layer normalization.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.TransformerDecoderLayer(
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
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        mem_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        mem_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool = False,
        mem_is_causal: bool = False
    ) -> Tensor

Computes the forward pass of the Transformer decoder layer.

**Inputs:**

- **tgt** (*Tensor*):
  The target input tensor of shape :math:`(N, L_t, d_{model})`.

- **memory** (*Tensor*):
  The encoder output tensor of shape :math:`(N, L_m, d_{model})`.

- **tgt_mask** (*Tensor | None*, optional):
  A mask of shape :math:`(L_t, L_t)` applied to self-attention weights.
  Default is `None`.

- **mem_mask** (*Tensor | None*, optional):
  A mask of shape :math:`(L_t, L_m)` applied to cross-attention weights.
  Default is `None`.

- **tgt_key_padding_mask** (*Tensor | None*, optional):
  A mask of shape :math:`(N, L_t)`, where non-zero values indicate positions that should be ignored.
  Default is `None`.

- **mem_key_padding_mask** (*Tensor | None*, optional):
  A mask of shape :math:`(N, L_m)`, where non-zero values indicate positions that should be ignored.
  Default is `None`.

- **tgt_is_causal** (*bool*, optional, default=False):
  If `True`, enforces a lower-triangular mask in self-attention.

- **mem_is_causal** (*bool*, optional, default=False):
  If `True`, enforces a lower-triangular mask in cross-attention.

**Output:**

- **Tensor**: The output tensor of shape :math:`(N, L_t, d_{model})`.

Mathematical Details
--------------------
The Transformer decoder layer consists of the following computations:

1. **Masked Multi-Head Self-Attention**
   
   .. math::

       A_{self} = \operatorname{softmax} \left( \frac{QK^T}{\sqrt{d_h}} + M_t \right) V
   
   where :math:`M_t` is the target mask.

2. **Multi-Head Cross-Attention**
   
   .. math::

       A_{cross} = \operatorname{softmax} \left( \frac{QK^T}{\sqrt{d_h}} + M_m \right) V
   
   where :math:`M_m` is the memory mask.

3. **Feedforward Network**
   
   .. math::

       F(x) = \operatorname{Activation}(x W_1 + b_1) W_2 + b_2
   
4. **Layer Normalization and Residual Connections**
   
   - If `norm_first=False`:
     
     .. math::

         y = \operatorname{LayerNorm}(x + A_{self})
         z = \operatorname{LayerNorm}(y + A_{cross})
         out = \operatorname{LayerNorm}(z + F(z))
   
   - If `norm_first=True`:
     
     .. math::

         y = x + A_{self}
         z = y + A_{cross}
         out = z + F(z)

Usage Example
-------------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    # Initialize TransformerDecoderLayer
    decoder_layer = nn.TransformerDecoderLayer(d_model=512, num_heads=8)

    # Create random input tensors
    tgt = lucid.random.randn(16, 10, 512)  # (batch, seq_len, embed_dim)
    memory = lucid.random.randn(16, 20, 512)  # Encoder output
    
    # Compute decoder output
    output = decoder_layer(tgt, memory)
    print(output.shape)  # Expected output: (16, 10, 512)

