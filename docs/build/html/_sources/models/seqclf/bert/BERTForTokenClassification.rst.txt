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
   :no-index:

Compute per-token classification logits for each sequence position.

.. automethod:: lucid.models.BERTForTokenClassification.get_loss
   :no-index:

Compute token classification loss with optional ignored indices.

.. automethod:: lucid.models.BERTForTokenClassification.predict_token_labels
   :no-index:

Return predicted token labels by argmax over class logits.

.. automethod:: lucid.models.BERTForTokenClassification.get_accuracy
   :no-index:

Compute token-level accuracy with optional ignored indices.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_token_classification_base(num_labels=2)
    >>> print(model)
    BERTForTokenClassification(...)

.. code-block:: python

    >>> logits = model(input_ids=input_ids, attention_mask=attention_mask)
    >>> loss = model.get_loss(labels=labels, input_ids=input_ids, attention_mask=attention_mask)
    >>> acc = model.get_accuracy(labels=labels, input_ids=input_ids, attention_mask=attention_mask)
