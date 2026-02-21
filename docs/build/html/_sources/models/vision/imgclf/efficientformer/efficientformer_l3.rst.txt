efficientformer_l3
==================

.. autofunction:: lucid.models.efficientformer_l3

The `efficientformer_l3` function instantiates the **L3** variant of 
the `EfficientFormer` architecture. This configuration offers a balance 
between computational efficiency and accuracy, suitable for mid-range devices and fast inference.

**Total Parameters**: 30,893,000

Function Signature
------------------

.. code-block:: python

    @register_model
    def efficientformer_l3(num_classes: int = 1000, **kwargs) -> EfficientFormer

Parameters
----------

- **num_classes** (*int*, optional):  
  Number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):  
  Additional keyword arguments to customize the `EfficientFormer` module.

Returns
-------

- **EfficientFormer**:  
  An instance of the `EfficientFormer` model configured with the L3 settings.

Specifications
--------------

- **depths**: (4, 4, 12, 6)  
- **embed_dims**: (64, 128, 320, 512)  
- **num_vit**: 2  

Examples
--------

**Creating a Default EfficientFormer-L3 Model**

.. code-block:: python

    import lucid.models as models

    model = models.efficientformer_l3()
    print(model)  # Displays the EfficientFormer-L3 architecture

**Custom Number of Classes**

.. code-block:: python

    model = models.efficientformer_l3(num_classes=5)
    print(model)
