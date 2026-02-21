Text Models
===========

.. toctree::
    :maxdepth: 1
    :hidden:

    Transformer <transformer/Transformer.rst>
    BERT <bert/BERT.rst>
    RoFormer <roformer/RoFormer.rst>

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
      - Pre-Trained

    * - :math:`\text{BERT}`
      - `BERT <bert/BERT>`_
      - :math:`(N,L)`
      - *Depends*
      - –

    * - :math:`\text{BERT}_\text{Pre}`
      - `BERTForPreTraining <bert/BERTForPreTraining>`_
      - :math:`(N,L)`
      - 110,106,428
      - ✅

    * - :math:`\text{BERT}_\text{MLM}`
      - `BERTForMaskedLM <bert/BERTForMaskedLM>`_
      - :math:`(N,L)`
      - 109,514,298
      - ❌

    * - :math:`\text{BERT}_\text{CLM}`
      - `BERTForCausalLM <bert/BERTForCausalLM>`_
      - :math:`(N,L)`
      - 109,514,298
      - ❌

    * - :math:`\text{BERT}_\text{NSP}`
      - `BERTForNextSentencePrediction <bert/BERTForNextSentencePrediction>`_
      - :math:`(N,L)`
      - 109,483,778
      - ❌

    * - :math:`\text{BERT}_\text{SC}`
      - `BERTForSequenceClassification <bert/BERTForSequenceClassification>`_
      - :math:`(N,L)`
      - 109,483,778
      - ❌

    * - :math:`\text{BERT}_\text{TC}`
      - `BERTForTokenClassification <bert/BERTForTokenClassification>`_
      - :math:`(N,L)`
      - 108,895,493
      - ❌

    * - :math:`\text{BERT}_\text{QA}`
      - `BERTForQuestionAnswering <bert/BERTForQuestionAnswering>`_
      - :math:`(N,L)`
      - 108,893,186
      - ❌

RoFormer
--------
|transformer-badge| |encoder-only-transformer-badge|

RoFormer is a rotary-position-embedding variant of BERT-style encoders,
retaining bidirectional Transformer blocks while replacing absolute positional
embedding usage in self-attention with RoPE.

 Su, Jianlin, et al. "RoFormer: Enhanced Transformer with Rotary Position
 Embedding." arXiv, 15 Apr. 2021, https://doi.org/10.48550/arXiv.2104.09864.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Pre-Trained

    * - :math:`\text{RoFormer}`
      - `RoFormer <roformer/RoFormer>`_
      - :math:`(N,L)`
      - *Depends*
      - –

    * - :math:`\text{RoFormer}_\text{MLM}`
      - `RoFormerForMaskedLM <roformer/RoFormerForMaskedLM>`_
      - :math:`(N,L)`
      - 109,711,674
      - ❌

    * - :math:`\text{RoFormer}_\text{SC}`
      - `RoFormerForSequenceClassification <roformer/RoFormerForSequenceClassification>`_
      - :math:`(N,L)`
      - 109,090,562
      - ❌

    * - :math:`\text{RoFormer}_\text{TC}`
      - `RoFormerForTokenClassification <roformer/RoFormerForTokenClassification>`_
      - :math:`(N,L)`
      - 109,090,562
      - ❌

    * - :math:`\text{RoFormer}_\text{MC}`
      - `RoFormerForMultipleChoice <roformer/RoFormerForMultipleChoice>`_
      - :math:`(N,C,L)`
      - 109,089,793
      - ❌

    * - :math:`\text{RoFormer}_\text{QA}`
      - `RoFormerForQuestionAnswering <roformer/RoFormerForQuestionAnswering>`_
      - :math:`(N,L)`
      - 109,090,562
      - ❌

.. note::

    Parameter counts are based on the smallest known variants for each model families.

*To be implemented...🔮*
