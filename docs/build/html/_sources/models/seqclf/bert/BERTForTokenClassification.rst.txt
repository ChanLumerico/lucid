BERTForTokenClassification
==========================

.. autoclass:: lucid.models.BERTForTokenClassification

The `BERTForTokenClassification` class predicts labels for each token
from sequence-level hidden states.

Class Signature
---------------

.. code-block:: python

    class BERTForTokenClassification(config: BERTConfig, num_labels: int = 2)

Parameters
----------
- **config** (*BERTConfig*): BERT configuration for token-level outputs.
- **num_labels** (*int*, optional): Number of target classes. Default is 2.

Methods
-------

.. automethod:: lucid.models.BERTForTokenClassification.forward

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_token_classification_base(num_labels=2)
    >>> print(model)
    BERTForTokenClassification(...)
