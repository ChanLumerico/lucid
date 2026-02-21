RoFormerForTokenClassification
==============================

.. autoclass:: lucid.models.RoFormerForTokenClassification

The `RoFormerForTokenClassification` class predicts labels for each token
from RoFormer hidden states.

Class Signature
---------------

.. code-block:: python

    class RoFormerForTokenClassification(config: RoFormerConfig, num_labels: int = 2)

Parameters
----------
- **config** (*RoFormerConfig*): RoFormer configuration for token-level outputs.
- **num_labels** (*int*, optional): Number of target classes. Default is 2.

Methods
-------

.. automethod:: lucid.models.RoFormerForTokenClassification.forward
   :no-index:

.. automethod:: lucid.models.RoFormerForTokenClassification.get_loss
   :no-index:

.. automethod:: lucid.models.RoFormerForTokenClassification.predict_token_labels
   :no-index:

.. automethod:: lucid.models.RoFormerForTokenClassification.get_accuracy
   :no-index:

.. automethod:: lucid.models.RoFormerForTokenClassification.get_loss_from_text
   :no-index:

.. automethod:: lucid.models.RoFormerForTokenClassification.predict_token_labels_from_text
   :no-index:

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> config = models.RoFormerConfig.base(vocab_size=50000)
    >>> model = models.RoFormerForTokenClassification(config, num_labels=5)
    >>> print(model)
    RoFormerForTokenClassification(...)
