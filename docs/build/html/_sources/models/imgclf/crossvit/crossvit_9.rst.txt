crossvit_9
==========

.. autofunction:: lucid.models.crossvit_9

The `crossvit_9` function provides a convenient way to create an instance of the 
`CrossViT` module with a lightweight configuration optimized for efficiency, 
following the 9-layer variant described in the CrossViT paper.

**Total Parameters**: 8,553,296

Function Signature
------------------

.. code-block:: python

    @register_model
    def crossvit_9(num_classes: int = 1000, **kwargs) -> CrossViT

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments to customize the `CrossViT` module.

Returns
-------

- **CrossViT**:
  An instance of the `CrossViT` module configured with the 9-layer variant settings.

Specifications
--------------

- **img_size**: [240, 224]
- **embed_dim**: [128, 256]
- **depth**: [[1, 3, 0], [1, 3, 0], [1, 3, 0]]
- **num_heads**: [4, 4]
- **mlp_ratio**: [3, 3, 1]
- **qkv_bias**: True

Examples
--------

**Creating a Default CrossViT-9 Model**

.. code-block:: python

    import lucid.models as models

    # Create a CrossViT-9 model with 1000 output classes
    model = models.crossvit_9()

    print(model)  # Displays the CrossViT-9 architecture

**Custom Number of Classes**

.. code-block:: python

    # Create a CrossViT-9 model with 10 output classes
    model = models.crossvit_9(num_classes=10)

    print(model)  # Displays the CrossViT-9 architecture with modified output