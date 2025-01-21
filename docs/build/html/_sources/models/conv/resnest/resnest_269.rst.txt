resnest_269
===========

.. autofunction:: lucid.models.resnest_269

The `resnest_269` function creates an instance of the ResNeSt-269 model, 
a lightweight variant of the ResNeSt architecture, tailored for tasks requiring 
fewer parameters.

**Total Parameters**: 110,929,224

Function Signature
------------------

.. code-block:: python

    @register_model
    def resnest_269(num_classes: int = 1000, **kwargs) -> ResNeSt


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
  An instance of the ResNeSt-269 model, configured with the provided parameters.

Layer Configuration
-------------------
The layer configuration for ResNeSt-269 is `[3, 30, 48, 8]`, 
which represents the number of blocks in each of the four stages of 
the ResNet architecture:

1. **Stage 1**: 3 block
2. **Stage 2**: 30 block
3. **Stage 3**: 48 block
4. **Stage 4**: 8 block

Examples
--------

.. code-block:: python

    from lucid.models import resnest_269

    # Create a ResNeSt-269 model for 10-class classification
    model = resnest_269(num_classes=10, base_width=64, stem_width=32)

    # Forward pass with a sample input
    input_tensor = lucid.random.randn((1, 3, 224, 224))
    output = model(input_tensor)
    print(output.shape)  # Output: (1, 10)
