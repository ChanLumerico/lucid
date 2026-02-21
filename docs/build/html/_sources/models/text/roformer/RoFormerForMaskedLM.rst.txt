RoFormerForMaskedLM
===================

.. autoclass:: lucid.models.RoFormerForMaskedLM

The `RoFormerForMaskedLM` class attaches a masked language modeling head
to the RoFormer backbone.

Class Signature
---------------

.. code-block:: python

    class RoFormerForMaskedLM(config: RoFormerConfig)

Parameters
----------
- **config** (*RoFormerConfig*): RoFormer configuration for masked language modeling.

Methods
-------

.. automethod:: lucid.models.RoFormerForMaskedLM.forward
   :no-index:

.. automethod:: lucid.models.RoFormerForMaskedLM.get_loss
   :no-index:

.. automethod:: lucid.models.RoFormerForMaskedLM.create_masked_lm_inputs
   :no-index:

.. automethod:: lucid.models.RoFormerForMaskedLM.predict_token_ids
   :no-index:

.. automethod:: lucid.models.RoFormerForMaskedLM.get_accuracy
   :no-index:

.. automethod:: lucid.models.RoFormerForMaskedLM.get_loss_from_text
   :no-index:

.. automethod:: lucid.models.RoFormerForMaskedLM.predict_token_ids_from_text
   :no-index:

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> config = models.RoFormerConfig.base(vocab_size=50000)
    >>> model = models.RoFormerForMaskedLM(config)
    >>> print(model)
    RoFormerForMaskedLM(...)
