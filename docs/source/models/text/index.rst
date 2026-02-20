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

    * - :math:`\text{BERT}_\text{base}`
      - `bert_for_pre_training_base <bert/bert_for_pre_training_base>`_
      - :math:`(N,L)`
      - 110,106,428
      - 28.50G

    * - :math:`\text{BERT}_\text{large}`
      - `bert_for_pre_training_large <bert/bert_for_pre_training_large>`_
      - :math:`(N,L)`
      - 336,226,108
      - 87.19G

    * - :math:`\text{BERT-MLM}_\text{base}`
      - `bert_for_masked_lm_base <bert/bert_for_masked_lm_base>`_
      - :math:`(N,L)`
      - 109,514,298
      - 28.50G

    * - :math:`\text{BERT-MLM}_\text{large}`
      - `bert_for_masked_lm_large <bert/bert_for_masked_lm_large>`_
      - :math:`(N,L)`
      - 335,174,458
      - 87.19G

    * - :math:`\text{BERT-CLM}_\text{base}`
      - `bert_for_causal_lm_base <bert/bert_for_causal_lm_base>`_
      - :math:`(N,L)`
      - 109,514,298
      - 28.50G

    * - :math:`\text{BERT-CLM}_\text{large}`
      - `bert_for_causal_lm_large <bert/bert_for_causal_lm_large>`_
      - :math:`(N,L)`
      - 335,174,458
      - 87.19G

    * - :math:`\text{BERT-NSP}_\text{base}`
      - `bert_for_next_sentence_prediction_base <bert/bert_for_next_sentence_prediction_base>`_
      - :math:`(N,L)`
      - 109,483,778
      - 22.35G

    * - :math:`\text{BERT-NSP}_\text{large}`
      - `bert_for_next_sentence_prediction_large <bert/bert_for_next_sentence_prediction_large>`_
      - :math:`(N,L)`
      - 335,143,938
      - 78.92G

    * - :math:`\text{BERT-SC}_\text{base}`
      - `bert_for_sequence_classification_base <bert/bert_for_sequence_classification_base>`_
      - :math:`(N,L)`
      - 109,483,778
      - 22.35G

    * - :math:`\text{BERT-SC}_\text{large}`
      - `bert_for_sequence_classification_large <bert/bert_for_sequence_classification_large>`_
      - :math:`(N,L)`
      - 335,143,938
      - 78.92G

    * - :math:`\text{BERT-TC}_\text{base}`
      - `bert_for_token_classification_base <bert/bert_for_token_classification_base>`_
      - :math:`(N,L)`
      - 108,895,493
      - 22.35G

    * - :math:`\text{BERT-TC}_\text{large}`
      - `bert_for_token_classification_large <bert/bert_for_token_classification_large>`_
      - :math:`(N,L)`
      - 334,097,413
      - 78.92G

    * - :math:`\text{BERT-QA}_\text{base}`
      - `bert_for_question_answering_base <bert/bert_for_question_answering_base>`_
      - :math:`(N,L)`
      - 108,893,186
      - 22.35G

    * - :math:`\text{BERT-QA}_\text{large}`
      - `bert_for_question_answering_large <bert/bert_for_question_answering_large>`_
      - :math:`(N,L)`
      - 334,094,338
      - 78.92G

*To be implemented...🔮*
