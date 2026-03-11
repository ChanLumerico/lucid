BERTForTokenClassificationConfig
================================

.. autoclass:: lucid.models.BERTForTokenClassificationConfig

`BERTForTokenClassificationConfig` stores the backbone `BERTConfig` and token
label count used by :class:`lucid.models.BERTForTokenClassification`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class BERTForTokenClassificationConfig:
        bert_config: BERTConfig
        num_labels: int = 2

Parameters
----------

- **bert_config** (*BERTConfig*):
  Backbone configuration for token-level predictions.
- **num_labels** (*int*):
  Number of token labels.

Validation
----------

- `bert_config` must be an instance of `BERTConfig`.
- `num_labels` must be greater than `0`.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.BERTForTokenClassificationConfig(
        bert_config=models.BERTConfig.base(add_pooling_layer=True),
        num_labels=5,
    )
    model = models.BERTForTokenClassification(config)
