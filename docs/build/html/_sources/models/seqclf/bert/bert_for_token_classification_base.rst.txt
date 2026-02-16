bert_for_token_classification_base
==================================

.. autofunction:: lucid.models.bert_for_token_classification_base

The `bert_for_token_classification_base` function creates a BERT-Base model with
a token-level classification head.

**Total Parameters**: 108,895,493

Function Signature
------------------

.. code-block:: python

    @register_model
    def bert_for_token_classification_base(
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
    >>> model = models.bert_for_token_classification_base(num_labels=2)
    >>> print(model)
    BERTForTokenClassification(...)
