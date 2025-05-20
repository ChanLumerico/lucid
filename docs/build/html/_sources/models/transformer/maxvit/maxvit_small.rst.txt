maxvit_small
============

.. autofunction:: lucid.models.maxvit_small

The `maxvit_small` function constructs a compact yet more expressive variant of the MaxViT model.  
Compared to `maxvit_tiny`, it increases channel width, enabling stronger 
representational capacity while maintaining efficiency.

Total Parameters: 55,757,304

Function Signature
------------------

.. code-block:: python

    def maxvit_small(
        in_channels: int = 3,
        num_classes: int = 1000,
        **kwargs
    ) -> MaxViT

Parameters
----------

- **in_channels** (*int*, optional):  
  Number of input image channels. Default is `3`.

- **num_classes** (*int*, optional):  
  Number of output classes for classification. Default is `1000`.

- **kwargs** (*any*):  
  Additional keyword arguments passed to the :class:`MaxViT` constructor.

Model Configuration
-------------------

This preset uses the following setup:

- **depths**: `(2, 2, 5, 2)`
- **channels**: `(96, 192, 384, 768)`
- **embed_dim**: `64`

Example
-------

.. code-block:: python

    import lucid
    from lucid.models.transformer import maxvit_small

    model = maxvit_small()
    input_tensor = lucid.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)  # (1, 1000)

.. tip::
    Use `**kwargs` to override components such as activation, norm layers, or dropout.
