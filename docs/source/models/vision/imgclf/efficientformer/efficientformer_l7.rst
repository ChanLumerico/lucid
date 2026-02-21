efficientformer_l7
==================

.. autofunction:: lucid.models.efficientformer_l7

The `efficientformer_l7` function constructs the **L7** configuration 
of the `EfficientFormer` model. It is the largest and most capable variant, 
designed for high-accuracy applications while remaining efficient enough 
for deployment on high-end edge devices.

**Total Parameters**: 81,460,328

Function Signature
------------------

.. code-block:: python

    @register_model
    def efficientformer_l7(num_classes: int = 1000, **kwargs) -> EfficientFormer

Parameters
----------

- **num_classes** (*int*, optional):  
  Number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):  
  Additional keyword arguments to customize the `EfficientFormer` module.

Returns
-------

- **EfficientFormer**:  
  An instance of the `EfficientFormer` model configured with the L7 settings.

Specifications
--------------

- **depths**: (6, 6, 18, 8)  
- **embed_dims**: (96, 192, 384, 768)  
- **num_vit**: 3  

Examples
--------

**Creating a Default EfficientFormer-L7 Model**

.. code-block:: python

    import lucid.models as models

    model = models.efficientformer_l7()
    print(model)  # Displays the EfficientFormer-L7 architecture

**Custom Number of Classes**

.. code-block:: python

    model = models.efficientformer_l7(num_classes=200)
    print(model)
