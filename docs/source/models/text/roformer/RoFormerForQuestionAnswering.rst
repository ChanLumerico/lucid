RoFormerForQuestionAnswering
============================

.. autoclass:: lucid.models.RoFormerForQuestionAnswering

The `RoFormerForQuestionAnswering` class predicts start and end logits for
extractive question answering.

Class Signature
---------------

.. code-block:: python

    class RoFormerForQuestionAnswering(config: RoFormerConfig)

Parameters
----------
- **config** (*RoFormerConfig*): RoFormer configuration for token span prediction.

Methods
-------

.. automethod:: lucid.models.RoFormerForQuestionAnswering.forward
   :no-index:

.. automethod:: lucid.models.RoFormerForQuestionAnswering.get_loss
   :no-index:

.. automethod:: lucid.models.RoFormerForQuestionAnswering.predict_spans
   :no-index:

.. automethod:: lucid.models.RoFormerForQuestionAnswering.get_best_spans
   :no-index:

.. automethod:: lucid.models.RoFormerForQuestionAnswering.get_accuracy
   :no-index:

.. automethod:: lucid.models.RoFormerForQuestionAnswering.predict_spans_from_text
   :no-index:

.. automethod:: lucid.models.RoFormerForQuestionAnswering.predict_answer_from_text
   :no-index:

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> config = models.RoFormerConfig.base(vocab_size=50000)
    >>> model = models.RoFormerForQuestionAnswering(config)
    >>> print(model)
    RoFormerForQuestionAnswering(...)
