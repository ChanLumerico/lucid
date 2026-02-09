bert_for_next_sentence_prediction_base
======================================

.. autofunction:: lucid.models.bert_for_next_sentence_prediction_base

The `bert_for_next_sentence_prediction_base` function creates a BERT-Base model
for binary next-sentence prediction.

**Total Parameters**: 109,483,778

Function Signature
------------------

.. code-block:: python

    @register_model
    def bert_for_next_sentence_prediction_base(
        vocab_size: int = 30522,
        **kwargs,
    ) -> BERTForNextSentencePrediction

Parameters
----------
- **vocab_size** (*int*, optional): Vocabulary size. Default is 30,522.
- **kwargs**: Additional keyword arguments forwarded to BERT configuration.

Returns
-------
- **BERTForNextSentencePrediction**:
  A model with NSP classification head.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_next_sentence_prediction_base()
    >>> print(model)
    BERTForNextSentencePrediction(...)
