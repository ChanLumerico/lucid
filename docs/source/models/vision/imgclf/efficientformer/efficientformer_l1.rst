efficientformer_l1
==================

.. autofunction:: lucid.models.efficientformer_l1

The `efficientformer_l1` function provides a convenient way to create an instance of the 
`EfficientFormer` module using the lightweight **L1** configuration, designed for high efficiency 
on mobile and edge devices while maintaining good accuracy.

**Total Parameters**: 11,840,928

Function Signature
------------------

.. code-block:: python

    @register_model
    def efficientformer_l1(num_classes: int = 1000, **kwargs) -> EfficientFormer

Parameters
----------

- **num_classes** (*int*, optional):  
  Number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):  
  Additional keyword arguments to customize the `EfficientFormer` module.

Returns
-------

- **EfficientFormer**:  
  An instance of the `EfficientFormer` model configured with the L1 settings.

Specifications
--------------

- **depths**: (3, 2, 6, 4)  
- **embed_dims**: (48, 96, 224, 448)  
- **num_vit**: 1

Examples
--------

**Creating a Default EfficientFormer-L1 Model**

.. code-block:: python

    import lucid.models as models

    # Create an EfficientFormer-L1 model with 1000 output classes
    model = models.efficientformer_l1()

    print(model)  # Displays the EfficientFormer-L1 architecture

**Custom Number of Classes**

.. code-block:: python

    # Create an EfficientFormer-L1 model with 10 output classes
    model = models.efficientformer_l1(num_classes=10)

    print(model)  # Displays the architecture with modified output classes