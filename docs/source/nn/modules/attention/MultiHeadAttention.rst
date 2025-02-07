nn.MultiHeadAttention
=====================

.. autoclass:: lucid.nn.MultiHeadAttention

Overview
--------
The `MultiHeadAttention` module implements multi-head attention, a key mechanism 
in transformer architectures. It projects the input queries, keys, and values into 
multiple subspaces (heads), applies scaled dot-product attention in parallel, 
and then concatenates and projects the results back to the embedding dimension.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.MultiHeadAttention(
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
    )

Parameters
----------
- **embed_dim** (*int*):  
  The total dimensionality of the input embeddings, denoted as :math:`d_{model}`.

- **num_heads** (*int*):  
  The number of attention heads, denoted as :math:`H`.
  .. warning::
    
     The embedding dimension :math:`d_{model}` must be divisible by :math:`H`, i.e.,  
     :math:`d_h = \frac{d_{model}}{H}` must be an integer.

- **dropout** (*float*, optional, default=0.0):  
  Dropout probability applied to the attention weights.

- **bias** (*bool*, optional, default=True):  
  If `True`, enables learnable bias terms in the linear projections.

- **add_bias_kv** (*bool*, optional, default=False):  
  If `True`, adds learnable bias vectors to the key and value sequences.

- **add_zero_attn** (*bool*, optional, default=False):  
  If `True`, appends an all-zero attention vector to the key and value sequences.

- **kdim** (*int* or *None*, optional):  
  The dimensionality of the key projections. Defaults to `embed_dim` if not specified.

- **vdim** (*int* or *None*, optional):  
  The dimensionality of the value projections. Defaults to `embed_dim` if not specified.

Mathematical Details
--------------------
The multi-head attention mechanism consists of the following computations:

1. **Linear Projections**
   
   Given input tensors:

   - Query: :math:`Q \in \mathbb{R}^{N \times L_q \times d_{model}}`
   - Key: :math:`K \in \mathbb{R}^{N \times L_k \times d_{model}}`
   - Value: :math:`V \in \mathbb{R}^{N \times L_v \times d_{model}}`

   They are projected using learnable weight matrices:

   .. math::

       Q' = Q W^Q, \quad K' = K W^K, \quad V' = V W^V

   where :math:`W^Q, W^K, W^V \in \mathbb{R}^{d_{model} \times d_{model}}`.

2. **Splitting into Multiple Heads**
   
   The projected tensors :math:`Q'`, :math:`K'`, and :math:`V'` are split into multiple heads:

   .. math::

       Q' \rightarrow \mathbb{R}^{N \times H \times L_q \times d_h}, \quad
       K' \rightarrow \mathbb{R}^{N \times H \times L_k \times d_h}, \quad
       V' \rightarrow \mathbb{R}^{N \times H \times L_v \times d_h},

   where :math:`d_h = \frac{d_{model}}{H}` is the per-head embedding size.

3. **Scaled Dot-Product Attention for Each Head**
   
   Each head computes attention independently:

   .. math::

       A_i = \operatorname{softmax} \left( \frac{Q_i K_i^\top}{\sqrt{d_h}} + M \right) V_i,

   where :math:`M` is an optional attention mask, and :math:`\frac{1}{\sqrt{d_h}}` is the scaling factor.

4. **Concatenation and Final Projection**
   
   The outputs from all heads are concatenated and projected:

   .. math::

       \text{Output} = \text{Concat}(A_1, \dots, A_H) W^O,

   where :math:`W^O \in \mathbb{R}^{d_{model} \times d_{model}}` is the final projection weight.

Additional Options
------------------
- **Bias for Keys and Values (add_bias_kv)**:  
  If enabled, learnable bias terms :math:`b_K, b_V \in \mathbb{R}^{1 \times 1 \times d_{model}}`  
  are added to keys and values:

  .. math::

      K = \text{Concat}(K, b_K), \quad V = \text{Concat}(V, b_V).

- **Zero Attention (add_zero_attn)**:  
  If enabled, an all-zero vector is appended:

  .. math::

      K = \text{Concat}(K, 0), \quad V = \text{Concat}(V, 0).

Usage Example
-------------
.. code-block:: python

    import lucid
    import lucid.nn as nn
    from lucid._tensor import Tensor

    # Initialize MultiHeadAttention with embedding dimension 512 and 8 heads
    mha = nn.MultiHeadAttention(embed_dim=512, num_heads=8)

    # Create random input tensors
    query = Tensor.randn(16, 10, 512)  # (batch, seq_len, embed_dim)
    key   = Tensor.randn(16, 10, 512)
    value = Tensor.randn(16, 10, 512)

    # Compute attention
    output = mha(query, key, value)
    print(output.shape)  # Expected output: (16, 10, 512)
