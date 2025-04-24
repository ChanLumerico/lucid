crossvit_15
===========

.. autofunction:: lucid.models.crossvit_15

The `crossvit_15` function provides a convenient way to create an instance of the 
`CrossViT` module with a medium-sized configuration, following the 15-layer variant 
described in the CrossViT paper.

**Total Parameters**: 27,528,464

Function Signature
------------------

.. code-block:: python

    @register_model
    def crossvit_15(num_classes: int = 1000, **kwargs) -> CrossViT

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments to customize the `CrossViT` module.

Returns
-------

- **CrossViT**:
  An instance of the `CrossViT` module configured with the 15-layer variant settings.

Specifications
--------------

- **img_size**: [240, 224]
- **embed_dim**: [192, 384]
- **depth**: [[1, 5, 0], [1, 5, 0], [1, 5, 0]]
- **num_heads**: [6, 6]
- **mlp_ratio**: [3, 3, 1]
- **qkv_bias**: True

Examples
--------

**Creating a Default CrossViT-15 Model**

.. code-block:: python

    import lucid.models as models

    # Create a CrossViT-15 model with 1000 output classes
    model = models.crossvit_15()

    print(model)  # Displays the CrossViT-15 architecture

**Custom Number of Classes**

.. code-block:: python

    # Create a CrossViT-15 model with 10 output classes
    model = models.crossvit_15(num_classes=10)

    print(model)  # Displays the CrossViT-15 architecture with modified output