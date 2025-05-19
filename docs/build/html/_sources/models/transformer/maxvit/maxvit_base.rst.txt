maxvit_base
===========

.. autofunction:: lucid.models.maxvit_base

The `maxvit_base` function constructs a high-capacity MaxViT model variant designed 
for more complex tasks, such as large-scale image classification. It offers deeper 
layers in the middle stages for enhanced expressiveness.

Total Parameters: 96,626,776

Function Signature
------------------

.. code-block:: python

    def maxvit_base(
        in_channels: int = 3,
        num_classes: int = 1000,
        **kwargs
    ) -> MaxViT

Parameters
----------

- **in_channels** (*int*, optional):  
  Number of input image channels. Default is `3`.

- **num_classes** (*int*, optional):  
  Number of classification categories. Default is `1000`.

- **kwargs** (*any*):  
  Additional keyword arguments passed to the :class:`MaxViT` constructor.

Model Configuration
-------------------

This base preset uses the following configuration:

- **depths**: `(2, 6, 14, 2)` — More blocks in the deeper stages.
- **channels**: `(96, 192, 384, 768)`
- **embed_dim**: `64`

Example
-------

.. code-block:: python

    import lucid
    from lucid.models.transformer import maxvit_base

    model = maxvit_base()
    input_tensor = lucid.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)  # (1, 1000)

.. note::
    The `maxvit_base` variant is suitable for high-resolution datasets and larger 
    model capacity benchmarks.

.. seealso::
    - :class:`lucid.models.transformer.MaxViT`
