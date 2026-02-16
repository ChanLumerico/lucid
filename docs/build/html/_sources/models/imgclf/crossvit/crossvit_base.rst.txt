crossvit_base
=============

.. autofunction:: lucid.models.crossvit_base

The `crossvit_base` function provides a convenient way to create an instance of the 
`CrossViT` module with a comprehensive configuration that balances depth and width, 
following the base variant described in the CrossViT paper.

**Total Parameters**: 105,025,232

Function Signature
------------------

.. code-block:: python

    @register_model
    def crossvit_base(num_classes: int = 1000, **kwargs) -> CrossViT

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments to customize the `CrossViT` module.

Returns
-------

- **CrossViT**:
  An instance of the `CrossViT` module configured with the base variant settings.

Specifications
--------------

- **img_size**: [240, 224]
- **embed_dim**: [384, 768]
- **depth**: [[1, 4, 0], [1, 4, 0], [1, 4, 0]]
- **num_heads**: [12, 12]
- **mlp_ratio**: [4, 4, 1]
- **qkv_bias**: True

Examples
--------

**Creating a Default CrossViT-Base Model**

.. code-block:: python

    import lucid.models as models

    # Create a CrossViT-Base model with 1000 output classes
    model = models.crossvit_base()

    print(model)  # Displays the CrossViT-Base architecture

**Custom Number of Classes**

.. code-block:: python

    # Create a CrossViT-Base model with 10 output classes
    model = models.crossvit_base(num_classes=10)

    print(model)  # Displays the CrossViT-Base architecture with modified output