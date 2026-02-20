Text Models
===========

.. toctree::
    :maxdepth: 1
    :hidden:

    Transformer <transformer/Transformer.rst>
    BERT <bert/BERT.rst>

Transformer
-----------
|transformer-badge|

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

BERT
----
|transformer-badge| |encoder-only-transformer-badge|

BERT is a Transformer-based model family for sequence understanding tasks,
including pre-training, language modeling, and sequence-level prediction heads.

 Devlin, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers 
 for Language Understanding." arXiv, 11 Oct. 2018, 
 https://doi.org/10.48550/arXiv.1810.04805. 

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs

    * - :math:`\text{BERT}`
      - `BERT <bert/BERT>`_
      - :math:`(N,L)`
      - *Depends*
      - *Depends*

    * - :math:`\text{BERT}_\text{Pre}`
      - `BERTForPreTraining <bert/BERTForPreTraining>`_
      - :math:`(N,L)`
      - 110,106,428
      - 28.50G

    * - :math:`\text{BERT-MLM}`
      - `BERTForMaskedLM <bert/BERTForMaskedLM>`_
      - :math:`(N,L)`
      - 109,514,298
      - 28.50G

    * - :math:`\text{BERT-CLM}`
      - `BERTForCausalLM <bert/BERTForCausalLM>`_
      - :math:`(N,L)`
      - 109,514,298
      - 28.50G

    * - :math:`\text{BERT-NSP}`
      - `BERTForNextSentencePrediction <bert/BERTForNextSentencePrediction>`_
      - :math:`(N,L)`
      - 109,483,778
      - 22.35G

    * - :math:`\text{BERT-SC}`
      - `BERTForSequenceClassification <bert/BERTForSequenceClassification>`_
      - :math:`(N,L)`
      - 109,483,778
      - 22.35G

    * - :math:`\text{BERT-TC}`
      - `BERTForTokenClassification <bert/BERTForTokenClassification>`_
      - :math:`(N,L)`
      - 108,895,493
      - 22.35G

    * - :math:`\text{BERT-QA}`
      - `BERTForQuestionAnswering <bert/BERTForQuestionAnswering>`_
      - :math:`(N,L)`
      - 108,893,186
      - 22.35G

*To be implemented...🔮*
