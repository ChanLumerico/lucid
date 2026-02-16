bert_for_sequence_classification_base
=====================================

.. autofunction:: lucid.models.bert_for_sequence_classification_base

The `bert_for_sequence_classification_base` function creates a BERT-Base model with
a sequence-level classification head.

**Total Parameters**: 109,483,778

Function Signature
------------------

.. code-block:: python

    @register_model
    def bert_for_sequence_classification_base(
        num_labels: int = 2,
        vocab_size: int = 30522,
        **kwargs,
    ) -> BERTForSequenceClassification

Parameters
----------
- **num_labels** (*int*, optional): Number of target classes. Default is 2.
- **vocab_size** (*int*, optional): Vocabulary size. Default is 30,522.
- **kwargs**: Additional keyword arguments forwarded to BERT configuration.

Returns
-------
- **BERTForSequenceClassification**:
  A model for sequence-level classification tasks.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_sequence_classification_base(num_labels=2)
    >>> print(model)
    BERTForSequenceClassification(...)
