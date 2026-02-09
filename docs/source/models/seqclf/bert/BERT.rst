BERT
====

.. toctree::
    :maxdepth: 1
    :hidden:

    BERTConfig.rst
    BERTForPreTraining.rst
    BERTForMaskedLM.rst
    BERTForCausalLM.rst
    BERTForNextSentencePrediction.rst
    BERTForSequenceClassification.rst
    BERTForTokenClassification.rst
    BERTForQuestionAnswering.rst
    bert_for_pre_training_base.rst
    bert_for_pre_training_large.rst
    bert_for_masked_lm_base.rst
    bert_for_masked_lm_large.rst
    bert_for_causal_lm_base.rst
    bert_for_causal_lm_large.rst
    bert_for_next_sentence_prediction_base.rst
    bert_for_next_sentence_prediction_large.rst
    bert_for_sequence_classification_base.rst
    bert_for_sequence_classification_large.rst
    bert_for_token_classification_base.rst
    bert_for_token_classification_large.rst
    bert_for_question_answering_base.rst
    bert_for_question_answering_large.rst

|transformer-badge| |encoder-only-transformer-badge| |seqclf-badge|

.. autoclass:: lucid.models.BERT

The `BERT` class provides a configurable bidirectional Transformer backbone
for text representation learning and can be paired with multiple task heads.

Class Signature
---------------

.. code-block:: python

    class BERT(config: BERTConfig)

Parameters
----------
- **config** (*BERTConfig*):
  Configuration object defining vocabulary, hidden dimensions, depth,
  attention setup, and runtime behavior (decoder/cache/pooling).

Methods
-------

.. automethod:: lucid.models.BERT.forward
.. automethod:: lucid.models.BERT.tie_weights
.. automethod:: lucid.models.BERT.get_input_embeddings
.. automethod:: lucid.models.BERT.set_input_embeddings
.. automethod:: lucid.models.BERT.get_output_embeddings
.. automethod:: lucid.models.BERT.set_output_embeddings

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> import lucid.models as models
    >>> config = models.BERTConfig(
    ...     vocab_size=30522,
    ...     hidden_size=768,
    ...     num_attention_heads=12,
    ...     num_hidden_layers=12,
    ...     intermediate_size=3072,
    ...     hidden_act=lucid.nn.functional.gelu,
    ...     hidden_dropout_prob=0.1,
    ...     attention_probs_dropout_prob=0.1,
    ...     max_position_embeddings=512,
    ...     tie_word_embedding=True,
    ...     type_vocab_size=2,
    ...     initializer_range=0.02,
    ...     layer_norm_eps=1e-12,
    ...     use_cache=False,
    ...     is_decoder=False,
    ...     add_cross_attention=False,
    ...     chunk_size_feed_forward=0,
    ... )
    >>> model = models.BERT(config)
    >>> print(model)
    BERT(...)
