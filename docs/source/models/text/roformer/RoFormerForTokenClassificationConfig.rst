RoFormerForTokenClassificationConfig
====================================

.. autoclass:: lucid.models.RoFormerForTokenClassificationConfig

`RoFormerForTokenClassificationConfig` stores the backbone `RoFormerConfig` and
token label count used by :class:`lucid.models.RoFormerForTokenClassification`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class RoFormerForTokenClassificationConfig:
        roformer_config: RoFormerConfig
        num_labels: int = 2

Parameters
----------

- **roformer_config** (*RoFormerConfig*):
  Backbone configuration for token-level predictions.
- **num_labels** (*int*):
  Number of token labels.

Validation
----------

- `roformer_config` must be an instance of `RoFormerConfig`.
- `num_labels` must be greater than `0`.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.RoFormerForTokenClassificationConfig(
        roformer_config=models.RoFormerConfig.base(vocab_size=50000),
        num_labels=5,
    )
    model = models.RoFormerForTokenClassification(config)
