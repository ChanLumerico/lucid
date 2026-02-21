RoFormerForSequenceClassification
=================================

.. autoclass:: lucid.models.RoFormerForSequenceClassification

The `RoFormerForSequenceClassification` class applies a classification head
to pooled RoFormer representations.

Class Signature
---------------

.. code-block:: python

    class RoFormerForSequenceClassification(config: RoFormerConfig, num_labels: int = 2)

Parameters
----------
- **config** (*RoFormerConfig*): RoFormer configuration with pooling enabled.
- **num_labels** (*int*, optional): Number of target classes. Default is 2.

Methods
-------

.. automethod:: lucid.models.RoFormerForSequenceClassification.forward
   :no-index:

.. automethod:: lucid.models.RoFormerForSequenceClassification.get_loss
   :no-index:

.. automethod:: lucid.models.RoFormerForSequenceClassification.predict_labels
   :no-index:

.. automethod:: lucid.models.RoFormerForSequenceClassification.predict_proba
   :no-index:

.. automethod:: lucid.models.RoFormerForSequenceClassification.get_accuracy
   :no-index:

.. automethod:: lucid.models.RoFormerForSequenceClassification.get_loss_from_text
   :no-index:

.. automethod:: lucid.models.RoFormerForSequenceClassification.predict_labels_from_text
   :no-index:

.. automethod:: lucid.models.RoFormerForSequenceClassification.predict_proba_from_text
   :no-index:

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> config = models.RoFormerConfig.base(vocab_size=50000)
    >>> model = models.RoFormerForSequenceClassification(config, num_labels=3)
    >>> print(model)
    RoFormerForSequenceClassification(...)
