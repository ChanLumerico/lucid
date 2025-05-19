crossvit_18
===========

.. autofunction:: lucid.models.crossvit_18

The `crossvit_18` function provides a convenient way to create an instance of the 
`CrossViT` module with a larger configuration, following the 18-layer variant 
described in the CrossViT paper.

**Total Parameters**: 43,271,408

Function Signature
------------------

.. code-block:: python

    @register_model
    def crossvit_18(num_classes: int = 1000, **kwargs) -> CrossViT

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments to customize the `CrossViT` module.

Returns
-------

- **CrossViT**:
  An instance of the `CrossViT` module configured with the 18-layer variant settings.

Specifications
--------------

- **img_size**: [240, 224]
- **embed_dim**: [224, 448]
- **depth**: [[1, 6, 0], [1, 6, 0], [1, 6, 0]]
- **num_heads**: [7, 7]
- **mlp_ratio**: [3, 3, 1]
- **qkv_bias**: True

Examples
--------

**Creating a Default CrossViT-18 Model**

.. code-block:: python

    import lucid.models as models

    # Create a CrossViT-18 model with 1000 output classes
    model = models.crossvit_18()

    print(model)  # Displays the CrossViT-18 architecture

**Custom Number of Classes**

.. code-block:: python

    # Create a CrossViT-18 model with 10 output classes
    model = models.crossvit_18(num_classes=10)

    print(model)  # Displays the CrossViT-18 architecture with modified output