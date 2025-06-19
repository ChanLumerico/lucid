Sequence-to-Sequence
====================

.. toctree::
    :maxdepth: 1
    :hidden:

    Transformer <transformer/Transformer.rst>


Transformer
-----------
|transformer-badge| |seq2seq-badge|

The Transformer is a deep learning architecture introduced by Vaswani et al. in 2017, 
designed for handling sequential data with self-attention mechanisms. It replaces 
traditional recurrent layers with attention-based mechanisms, enabling highly 
parallelized training and capturing long-range dependencies effectively.

 Vaswani, Ashish, et al. "Attention Is All You Need." 
 *Advances in Neural Information Processing Systems*, 2017, pp. 5998-6008.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs
    
    * - Transformer-Base
      - `transformer_base <transformer/transformer_base>`_
      - :math:`(N, L_{src})`, :math:`(N, L_{tgt})`
      - 62,584,544
      - :math:`O(N \cdot d_{m} \cdot L_{src} \cdot L_{tgt})`
    
    * - Transformer-Big
      - `transformer_big <transformer/transformer_big>`_
      - :math:`(N, L_{src})`, :math:`(N, L_{tgt})`
      - 213,237,472
      - :math:`O(N \cdot d_{m} \cdot L_{src} \cdot L_{tgt})`

*To be implemented...🔮*
