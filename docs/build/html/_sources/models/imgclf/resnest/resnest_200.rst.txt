resnest_200
===========

.. autofunction:: lucid.models.resnest_200

The `resnest_200` function creates an instance of the ResNeSt-200 model, 
a lightweight variant of the ResNeSt architecture, tailored for tasks requiring 
fewer parameters.

**Total Parameters**: 70,201,288

Function Signature
------------------

.. code-block:: python

    @register_model
    def resnest_200(num_classes: int = 1000, **kwargs) -> ResNeSt


Parameters
----------
- **num_classes** (*int*, optional):
  Number of output classes for the classification task. Defaults to 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments passed to the `ResNeSt` constructor, 
  allowing customization of the model's hyperparameters such as 
  `base_width`, `stem_width`, `cardinality`, and `radix`.

Returns
-------
- **ResNeSt**:
  An instance of the ResNeSt-200 model, configured with the provided parameters.

Layer Configuration
-------------------
The layer configuration for ResNeSt-200 is `[3, 24, 36, 3]`, 
which represents the number of blocks in each of the four stages of 
the ResNet architecture:

1. **Stage 1**: 3 block
2. **Stage 2**: 24 block
3. **Stage 3**: 36 block
4. **Stage 4**: 3 block

Examples
--------

.. code-block:: python

    from lucid.models import resnest_200

    # Create a ResNeSt-200 model for 10-class classification
    model = resnest_200(num_classes=10, base_width=64, stem_width=32)

    # Forward pass with a sample input
    input_tensor = lucid.random.randn((1, 3, 224, 224))
    output = model(input_tensor)
    print(output.shape)  # Output: (1, 10)
