BERTForSequenceClassificationConfig
===================================

.. autoclass:: lucid.models.BERTForSequenceClassificationConfig

`BERTForSequenceClassificationConfig` stores the backbone `BERTConfig` and
sequence label count used by :class:`lucid.models.BERTForSequenceClassification`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class BERTForSequenceClassificationConfig:
        bert_config: BERTConfig
        num_labels: int = 2

Parameters
----------

- **bert_config** (*BERTConfig*):
  Backbone configuration. Pooling should remain enabled.
- **num_labels** (*int*):
  Number of target classes.

Validation
----------

- `bert_config` must be an instance of `BERTConfig`.
- `num_labels` must be greater than `0`.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.BERTForSequenceClassificationConfig(
        bert_config=models.BERTConfig.base(add_pooling_layer=True),
        num_labels=2,
    )
    model = models.BERTForSequenceClassification(config)
