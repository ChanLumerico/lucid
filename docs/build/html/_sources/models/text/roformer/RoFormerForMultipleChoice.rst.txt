RoFormerForMultipleChoice
=========================

.. autoclass:: lucid.models.RoFormerForMultipleChoice

The `RoFormerForMultipleChoice` class scores candidate choices by flattening
`[batch, num_choices, seq]` inputs, running RoFormer encoding, and reshaping
logits back to `[batch, num_choices]`.

Class Signature
---------------

.. code-block:: python

    class RoFormerForMultipleChoice(config: RoFormerConfig)

Parameters
----------
- **config** (*RoFormerConfig*): RoFormer configuration with pooling enabled.

Methods
-------

.. automethod:: lucid.models.RoFormerForMultipleChoice.forward
   :no-index:

.. automethod:: lucid.models.RoFormerForMultipleChoice.get_loss
   :no-index:

.. automethod:: lucid.models.RoFormerForMultipleChoice.predict_labels
   :no-index:

.. automethod:: lucid.models.RoFormerForMultipleChoice.predict_proba
   :no-index:

.. automethod:: lucid.models.RoFormerForMultipleChoice.get_accuracy
   :no-index:

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> config = models.RoFormerConfig.base(vocab_size=50000)
    >>> model = models.RoFormerForMultipleChoice(config)
    >>> print(model)
    RoFormerForMultipleChoice(...)
