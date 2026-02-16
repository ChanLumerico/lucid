resnest_50_4s2x40d
==================

.. autofunction:: lucid.models.resnest_50_4s2x40d

The `resnest_50_4s2x40d` function creates an instance of the ResNeSt-50 
variant configured with specific hyperparameters (`4s2x40d`), which enhance its 
representational capabilities for image recognition tasks.

**Total Parameters**: 30,417,464

Function Signature
------------------

.. code-block:: python

    @register_model
    def resnest_50_4s2x40d(num_classes: int = 1000, **kwargs) -> ResNeSt

Parameters
----------
- **num_classes** (*int*, optional):
  Number of output classes for the classification task. Defaults to 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments passed to the `ResNeSt` constructor, 
  allowing further customization of the model.

Returns
-------
- **ResNeSt**:
  An instance of the ResNeSt-50 model, configured with the provided 
  parameters and `4s2x40d` settings.

Hyperparameter Configuration
----------------------------
- **4s**: Indicates 4 splits in the Split-Attention mechanism (radix = 4), 
  allowing the model to compute attention weights over 4 groups.

- **2x**: Specifies a cardinality of 2, meaning each group processes 
  a subset of channels with two separate convolution operations.

- **40d**: Denotes a base width of 40, which scales the number of channels 
  for intermediate feature maps.

Examples
--------

.. code-block:: python

    from lucid.models import resnest_50_4s2x40d

    # Create a ResNeSt-50 model for 10-class classification
    model = resnest_50_4s2x40d(num_classes=10)

    # Forward pass with a sample input
    input_tensor = lucid.random.randn((1, 3, 224, 224))
    output = model(input_tensor)
    print(output.shape)  # Output: (1, 10)
