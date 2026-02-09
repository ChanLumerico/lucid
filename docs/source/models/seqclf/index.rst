Sequence Classification
=======================

.. toctree::
    :maxdepth: 1
    :hidden:

    BERT <bert/BERT.rst>

BERT
----
|transformer-badge| |encoder-only-transformer-badge| |seqclf-badge|

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
      - Parameter Count
      - FLOPs

    * - BERT
      - `BERT <bert/BERT>`_
      - *Depends*
      - *Depends*

    * - bert_for_pre_training_base
      - `bert_for_pre_training_base <bert/bert_for_pre_training_base>`_
      - 110,106,428
      - 28.50G

    * - bert_for_pre_training_large
      - `bert_for_pre_training_large <bert/bert_for_pre_training_large>`_
      - 336,226,108
      - 87.19G

    * - bert_for_masked_lm_base
      - `bert_for_masked_lm_base <bert/bert_for_masked_lm_base>`_
      - 109,514,298
      - 28.50G

    * - bert_for_masked_lm_large
      - `bert_for_masked_lm_large <bert/bert_for_masked_lm_large>`_
      - 335,174,458
      - 87.19G

    * - bert_for_causal_lm_base
      - `bert_for_causal_lm_base <bert/bert_for_causal_lm_base>`_
      - 109,514,298
      - 28.50G

    * - bert_for_causal_lm_large
      - `bert_for_causal_lm_large <bert/bert_for_causal_lm_large>`_
      - 335,174,458
      - 87.19G

    * - bert_for_next_sentence_prediction_base
      - `bert_for_next_sentence_prediction_base <bert/bert_for_next_sentence_prediction_base>`_
      - 109,483,778
      - 22.35G

    * - bert_for_next_sentence_prediction_large
      - `bert_for_next_sentence_prediction_large <bert/bert_for_next_sentence_prediction_large>`_
      - 335,143,938
      - 78.92G

    * - bert_for_sequence_classification_base
      - `bert_for_sequence_classification_base <bert/bert_for_sequence_classification_base>`_
      - 109,483,778
      - 22.35G

    * - bert_for_sequence_classification_large
      - `bert_for_sequence_classification_large <bert/bert_for_sequence_classification_large>`_
      - 335,143,938
      - 78.92G

    * - bert_for_token_classification_base
      - `bert_for_token_classification_base <bert/bert_for_token_classification_base>`_
      - 108,895,493
      - 22.35G

    * - bert_for_token_classification_large
      - `bert_for_token_classification_large <bert/bert_for_token_classification_large>`_
      - 334,097,413
      - 78.92G

    * - bert_for_question_answering_base
      - `bert_for_question_answering_base <bert/bert_for_question_answering_base>`_
      - 108,893,186
      - 22.35G

    * - bert_for_question_answering_large
      - `bert_for_question_answering_large <bert/bert_for_question_answering_large>`_
      - 334,094,338
      - 78.92G
