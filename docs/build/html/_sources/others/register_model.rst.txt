lucid.register_model
====================

.. autofunction:: lucid.register_model

The `register_model` decorator registers a model function and its metadata into a JSON registry file. 
This is particularly useful for managing and organizing models in the `lucid` library.

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
3. The model's name, family (class name), parameter size, and architecture are extracted.
4. If the model's name already exists in the registry, the model is returned without making changes to the registry.
5. Otherwise, the new model's metadata is added to the registry and saved back to the JSON file.

Metadata Captured
-----------------
The following metadata is registered:

- **name** (*str*): Name of the function that creates the model.
- **family** (*str*): The class name of the instantiated model.
- **param_size** (*int*): The parameter size of the model.
- **arch** (*str*): The architecture's name derived from the package structure, 
  removing the `lucid.models.` prefix.

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
