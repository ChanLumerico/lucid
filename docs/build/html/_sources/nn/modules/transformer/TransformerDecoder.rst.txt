nn.TransformerDecoder
=====================

.. autoclass:: lucid.nn.TransformerDecoder

Overview
--------
The `TransformerDecoder` module stacks multiple `TransformerDecoderLayer` 
instances to form a complete Transformer decoder. It sequentially processes 
the target input through multiple decoder layers while attending to the 
encoder memory output. An optional layer normalization can be applied to 
the final output.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.TransformerDecoder(
        decoder_layer: TransformerDecoderLayer | nn.Module,
        num_layers: int,
        norm: nn.Module | None = None,
    )

Parameters
----------
- **decoder_layer** (*TransformerDecoderLayer | nn.Module*):
  A single instance of `TransformerDecoderLayer` that will be replicated for 
  `num_layers` times.

- **num_layers** (*int*):
  The number of decoder layers in the stack.

- **norm** (*nn.Module | None*, optional):
  An optional layer normalization module applied to the final output. 
  Default is `None`.

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

Computes the forward pass of the Transformer decoder.

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
  A mask of shape :math:`(N, L_t)`, where non-zero values indicate 
  positions that should be ignored. Default is `None`.

- **mem_key_padding_mask** (*Tensor | None*, optional):
  A mask of shape :math:`(N, L_m)`, where non-zero values indicate 
  positions that should be ignored. Default is `None`.

- **tgt_is_causal** (*bool*, optional, default=False):
  If `True`, enforces a lower-triangular mask in self-attention.

- **mem_is_causal** (*bool*, optional, default=False):
  If `True`, enforces a lower-triangular mask in cross-attention.

**Output:**

- **Tensor**: The output tensor of shape :math:`(N, L_t, d_{model})`.

Mathematical Details
--------------------
The Transformer decoder processes input through a sequence of decoder layers as follows:

1. **Iterative Decoding**
   
   Each target tensor :math:`T` is passed through `num_layers` decoder 
   layers while attending to the encoder memory:
   
   .. math::

       T_0 = T
       T_{i+1} = \operatorname{DecoderLayer}(T_i, M), \quad \forall i 
       \in [0, \text{num\_layers}-1]
   
   where :math:`M` represents the memory from the encoder.

2. **Optional Normalization**
   
   If `norm` is provided, it is applied to the final output:
   
   .. math::

       Y = \operatorname{LayerNorm}(T_{\text{num\_layers}})
   
   Otherwise, the final decoder layer output is returned.

Usage Example
-------------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    # Create a decoder layer
    decoder_layer = nn.TransformerDecoderLayer(d_model=512, num_heads=8)
    
    # Stack multiple decoder layers into a Transformer decoder
    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

    # Create random input tensors
    tgt = lucid.random.randn(16, 10, 512)  # (batch, seq_len, embed_dim)
    memory = lucid.random.randn(16, 20, 512)  # Encoder output
    
    # Compute decoder output
    output = transformer_decoder(tgt, memory)
    print(output.shape)  # Expected output: (16, 10, 512)

