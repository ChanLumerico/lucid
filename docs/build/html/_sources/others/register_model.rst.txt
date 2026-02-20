lucid.register_model
====================

.. autofunction:: lucid.register_model

The `register_model` decorator registers a model factory and its metadata into a JSON
registry file.

Function Signature
------------------

.. code-block:: python

    def register_model(func: Callable) -> Callable

Parameters
----------

- **func** (*Callable*):
  The function that instantiates and returns a model class instance.

Description
-----------

When a function decorated with `@register_model` is called, the following occurs:

1. The registry file (JSON) is loaded from `REGISTRY_PATH`.
2. The function is invoked to instantiate the model.
3. The model's parameter size and submodule count are extracted.
4. A hierarchical key path is derived from the model file location under
   `lucid/models` (for example, `lucid/models/vision/dense.py` ->
   `vision -> dense -> densenet_121`).
5. If the model entry already exists, the model is returned without modifying the
   registry.
6. Otherwise, the new model metadata is inserted and the registry is saved.

Metadata Captured
-----------------
Each model factory entry stores:

- **parameter_size** (*int*): Total number of model parameters.
- **submodule_count** (*int*): Total number of child submodules.

Examples
--------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 100)
            self.layer2 = nn.linear(100, 10)

    @register_model
    def my_simple_model(*args, **kwargs) -> SimpleModel:
        return SimpleModel(*args, **kwargs)

    model = my_simple_model()

This example shows a function `my_simple_model` that creates an instance of a 
`SimpleModel` class. When the function is called, the model's metadata is 
registered into the JSON registry file.
