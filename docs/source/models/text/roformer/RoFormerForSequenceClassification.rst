RoFormerForSequenceClassification
=================================

.. toctree::
    :maxdepth: 1
    :hidden:

    RoFormerForSequenceClassificationConfig.rst

.. autoclass:: lucid.models.RoFormerForSequenceClassification

The `RoFormerForSequenceClassification` class applies a classification head
to pooled RoFormer representations.

Class Signature
---------------

.. code-block:: python

    class RoFormerForSequenceClassification(config: RoFormerForSequenceClassificationConfig)

Parameters
----------
- **config** (*RoFormerForSequenceClassificationConfig*):
  Wrapper configuration containing the pooled RoFormer backbone config and
  the target class count.

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
    >>> config = models.RoFormerForSequenceClassificationConfig(
    ...     roformer_config=models.RoFormerConfig.base(vocab_size=50000),
    ...     num_labels=3,
    ... )
    >>> model = models.RoFormerForSequenceClassification(config)
    >>> print(model)
    RoFormerForSequenceClassification(...)
