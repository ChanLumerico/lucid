RoFormerForSequenceClassificationConfig
=======================================

.. autoclass:: lucid.models.RoFormerForSequenceClassificationConfig

`RoFormerForSequenceClassificationConfig` stores the backbone `RoFormerConfig`
and sequence label count used by :class:`lucid.models.RoFormerForSequenceClassification`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class RoFormerForSequenceClassificationConfig:
        roformer_config: RoFormerConfig
        num_labels: int = 2

Parameters
----------

- **roformer_config** (*RoFormerConfig*):
  Backbone configuration. Pooling should remain enabled.
- **num_labels** (*int*):
  Number of target classes.

Validation
----------

- `roformer_config` must be an instance of `RoFormerConfig`.
- `num_labels` must be greater than `0`.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.RoFormerForSequenceClassificationConfig(
        roformer_config=models.RoFormerConfig.base(vocab_size=50000),
        num_labels=3,
    )
    model = models.RoFormerForSequenceClassification(config)
