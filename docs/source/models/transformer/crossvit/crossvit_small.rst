crossvit_small
==============

.. autofunction:: lucid.models.crossvit_small

The `crossvit_small` function provides a convenient way to create an instance of the 
`CrossViT` module with a specific configuration optimized for a balance between model size 
and performance, following the small variant described in the CrossViT paper.

**Total Parameters**: 26,856,272

Function Signature
------------------

.. code-block:: python

    @register_model
    def crossvit_small(num_classes: int = 1000, **kwargs) -> CrossViT

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments to customize the `CrossViT` module.

Returns
-------

- **CrossViT**:
  An instance of the `CrossViT` module configured with the small variant settings.

Specifications
--------------

- **img_size**: [240, 224]
- **embed_dim**: [192, 384]
- **depth**: [[1, 4, 0], [1, 4, 0], [1, 4, 0]]
- **num_heads**: [6, 6]
- **mlp_ratio**: [4, 4, 1]
- **qkv_bias**: True

Examples
--------

**Creating a Default CrossViT-Small Model**

.. code-block:: python

    import lucid.models as models

    # Create a CrossViT-Small model with 1000 output classes
    model = models.crossvit_small()

    print(model)  # Displays the CrossViT-Small architecture

**Custom Number of Classes**

.. code-block:: python

    # Create a CrossViT-Small model with 10 output classes
    model = models.crossvit_small(num_classes=10)

    print(model)  # Displays the CrossViT-Small architecture with modified output