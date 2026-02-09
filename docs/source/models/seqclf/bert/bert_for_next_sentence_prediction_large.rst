bert_for_next_sentence_prediction_large
=======================================

.. autofunction:: lucid.models.bert_for_next_sentence_prediction_large

The `bert_for_next_sentence_prediction_large` function creates a BERT-Large model
for binary next-sentence prediction.

**Total Parameters**: 335,143,938

Function Signature
------------------

.. code-block:: python

    @register_model
    def bert_for_next_sentence_prediction_large(
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
    >>> model = models.bert_for_next_sentence_prediction_large()
    >>> print(model)
    BERTForNextSentencePrediction(...)
