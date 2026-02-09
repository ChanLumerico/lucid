bert_for_token_classification_large
===================================

.. autofunction:: lucid.models.bert_for_token_classification_large

The `bert_for_token_classification_large` function creates a BERT-Large model with
a token-level classification head.

**Total Parameters**: 334,097,413
**Total FLOPs**: 78.92G

Function Signature
------------------

.. code-block:: python

    @register_model
    def bert_for_token_classification_large(
        num_labels: int = 2,
        vocab_size: int = 30522,
        **kwargs,
    ) -> BERTForTokenClassification

Parameters
----------
- **num_labels** (*int*, optional): Number of token labels. Default is 2.
- **vocab_size** (*int*, optional): Vocabulary size. Default is 30,522.
- **kwargs**: Additional keyword arguments forwarded to BERT configuration.

Returns
-------
- **BERTForTokenClassification**:
  A model for token-level prediction tasks.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_token_classification_large(num_labels=2)
    >>> print(model)
    BERTForTokenClassification(...)
