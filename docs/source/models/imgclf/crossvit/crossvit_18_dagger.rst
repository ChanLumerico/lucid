crossvit_18_dagger
==================

.. autofunction:: lucid.models.crossvit_18_dagger

The `crossvit_18_dagger` function provides a convenient way to create an instance of the 
`CrossViT` module with an enhanced 18-layer configuration, following the dagger 
(:math:`^\dagger`) variant described in the CrossViT paper.

**Total Parameters**: 44,266,976

Function Signature
------------------

.. code-block:: python

    @register_model
    def crossvit_18_dagger(num_classes: int = 1000, **kwargs) -> CrossViT

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments to customize the `CrossViT` module.

Returns
-------

- **CrossViT**:
  An instance of the `CrossViT` module configured with the 18-layer dagger variant settings.

Specifications
--------------

- **img_size**: [240, 224]
- **embed_dim**: [224, 448]
- **depth**: [[1, 6, 0], [1, 6, 0], [1, 6, 0]]
- **num_heads**: [7, 7]
- **mlp_ratio**: [3, 3, 1]
- **qkv_bias**: True
- **multi_conv**: True

About Dagger (:math:`^\dagger`) Variants
----------------------------------------

The dagger (:math:`^\dagger`) variant, as introduced in the CrossViT paper, 
refers to models that use an enhanced patch embedding process. 

Instead of a single convolutional layer for patch embedding, these models employ 
multiple convolutional layers in a hierarchical fashion. Specifically:

- Standard variants use a single large convolution to project image patches 
  to embedding space

- Dagger variants use multiple smaller convolutions with stride=2, 
  creating a more CNN-like hierarchical feature extraction

- This hierarchical approach helps the model capture both global patterns 
  and local details more effectively

- The multi-scale approach provides better inductive bias for image processing tasks, 
  improving performance particularly for smaller models

The multi-convolution approach is activated by setting the `multi_conv=True` parameter, 
which replaces the standard patch embedding with a sequence of smaller convolutional 
operations, similar to the approach used in ResNet and other CNN architectures.

Examples
--------

**Creating a Default CrossViT-18-Dagger Model**

.. code-block:: python

    import lucid.models as models

    # Create a CrossViT-18-Dagger model with 1000 output classes
    model = models.crossvit_18_dagger()

    print(model)  # Displays the CrossViT-18-Dagger architecture

**Custom Number of Classes**

.. code-block:: python

    # Create a CrossViT-18-Dagger model with 10 output classes
    model = models.crossvit_18_dagger(num_classes=10)

    print(model)  # Displays the CrossViT-18-Dagger architecture with modified output