RoFormerForTokenClassification
==============================

.. toctree::
    :maxdepth: 1
    :hidden:

    RoFormerForTokenClassificationConfig.rst

.. autoclass:: lucid.models.RoFormerForTokenClassification

The `RoFormerForTokenClassification` class predicts labels for each token
from RoFormer hidden states.

Class Signature
---------------

.. code-block:: python

    class RoFormerForTokenClassification(config: RoFormerForTokenClassificationConfig)

Parameters
----------
- **config** (*RoFormerForTokenClassificationConfig*):
  Wrapper configuration containing the RoFormer backbone config and token label count.

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
    >>> config = models.RoFormerForTokenClassificationConfig(
    ...     roformer_config=models.RoFormerConfig.base(vocab_size=50000),
    ...     num_labels=5,
    ... )
    >>> model = models.RoFormerForTokenClassification(config)
    >>> print(model)
    RoFormerForTokenClassification(...)
