crossvit_tiny
=============

.. autofunction:: lucid.models.crossvit_tiny

The `crossvit_tiny` function provides a convenient way to create an instance of the 
`CrossViT` module with a specific configuration optimized for smaller model size while 
maintaining performance, following the tiny variant described in the CrossViT paper.

**Total Parameters**: 7,014,800

Function Signature
------------------

.. code-block:: python

    @register_model
    def crossvit_tiny(num_classes: int = 1000, **kwargs) -> CrossViT

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments to customize the `CrossViT` module.

Returns
-------

- **CrossViT**:
  An instance of the `CrossViT` module configured with the tiny variant settings.

Specifications
--------------

- **img_size**: [240, 224]
- **embed_dim**: [96, 192]
- **depth**: [[1, 4, 0], [1, 4, 0], [1, 4, 0]]
- **num_heads**: [3, 3]
- **mlp_ratio**: [4, 4, 1]
- **qkv_bias**: True

Examples
--------

**Creating a Default CrossViT-Tiny Model**

.. code-block:: python

    import lucid.models as models

    # Create a CrossViT-Tiny model with 1000 output classes
    model = models.crossvit_tiny()

    print(model)  # Displays the CrossViT-Tiny architecture

**Custom Number of Classes**

.. code-block:: python

    # Create a CrossViT-Tiny model with 10 output classes
    model = models.crossvit_tiny(num_classes=10)

    print(model)  # Displays the CrossViT-Tiny architecture with modified output