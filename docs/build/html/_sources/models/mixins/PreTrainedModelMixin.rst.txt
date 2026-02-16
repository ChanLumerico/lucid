PreTrainedModelMixin
====================

`PreTrainedModelMixin` is a small utility mixin for model classes that inherit from
`lucid.nn.Module`. It adds a chainable `from_pretrained(...)` method that resolves a
weight entry and applies it to the current model instance.

.. warning::

   Not all pre-built model base classes under `lucid.models` currently inherit
   `PreTrainedModelMixin`. As a result, `from_pretrained(...)` is not guaranteed
   to exist on every model class.

.. important::

   The model configuration used to instantiate the class must match the
   configuration expected by the pretrained weights you load with
   `from_pretrained(...)` (for example, hidden size, number of layers/heads,
   vocabulary size, and other structural settings). If these do not match,
   state dict loading may fail (or require `strict=False` with partial loading).

.. autoclass:: lucid.models.base.PreTrainedModelMixin

Class Signature
---------------

.. code-block:: python

    class PreTrainedModelMixin:
        def from_pretrained(self, weights: WeightEntry, strict: bool = True) -> Self

Parameters
----------

- **weights** (`WeightEntry`): A registered weight entry (for example `BERT_Weights.PRE_TRAIN_BASE`).
- **strict** (`bool`, default `True`): Passed through to `model.load_state_dict(..., strict=strict)`.
  Set `False` when loading a backbone into a different task head.

Returns
-------

- **self**: The same model instance, so calls can be chained.

Behavior
--------

- Works only on classes that are also `lucid.nn.Module`.
- Raises `RuntimeError` if called on a non-module object.
- Internally calls `lucid.weights.apply(...)`.

Method
------

.. automethod:: lucid.models.base.PreTrainedModelMixin.from_pretrained

Examples
--------

Basic pretrained load
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from lucid.models import SomeModel
    from lucid.weights import SomeModel_Weights

    model = SomeModel()
    model = model.from_pretrained(SomeModel_Weights.DEFAULT)
    model.eval()

Backbone transfer (`strict=False`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from lucid.models import TargetTaskModel
    from lucid.weights import SourceBackbone_Weights

    model = TargetTaskModel()
    model.from_pretrained(
        SourceBackbone_Weights.DEFAULT,
        strict=False,  # allow task-head key mismatch
    )

Chaining usage
~~~~~~~~~~~~~~

.. code-block:: python

    from lucid.models import SomeModel
    from lucid.weights import SomeModel_Weights

    model = (
        SomeModel()
        .from_pretrained(SomeModel_Weights.DEFAULT)
        .to("gpu")
        .eval()
    )

Custom model using the mixin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import lucid.nn as nn
    from lucid.models.base import PreTrainedModelMixin

    class MyModel(PreTrainedModelMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(128, 128)
