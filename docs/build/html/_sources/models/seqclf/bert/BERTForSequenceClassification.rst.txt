BERTForSequenceClassification
=============================

.. autoclass:: lucid.models.BERTForSequenceClassification

The `BERTForSequenceClassification` class applies a classification head
to pooled BERT representations.

Class Signature
---------------

.. code-block:: python

    class BERTForSequenceClassification(config: BERTConfig, num_labels: int = 2)

Parameters
----------
- **config** (*BERTConfig*): BERT configuration with pooling enabled.
- **num_labels** (*int*, optional): Number of target classes. Default is 2.

Methods
-------

.. automethod:: lucid.models.BERTForSequenceClassification.forward

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_sequence_classification_base(num_labels=2)
    >>> print(model)
    BERTForSequenceClassification(...)
