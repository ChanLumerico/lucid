efficientdet_d7
===============

.. autofunction:: lucid.models.efficientdet_d7

The `efficientdet_d7` function returns an instance of the `EfficientDet` model,  
configured with **compound coefficient 7**, the largest and most powerful model  
in the EfficientDet family.

**Total Parameters**: 87,173,148

Function Signature
------------------
.. code-block:: python

    def efficientdet_d7(num_classes: int = 80, **kwargs) -> EfficientDet

Parameters
----------
- **num_classes** (*int*, default=80):  
  Number of target object classes to detect.

- **kwargs**:  
  Additional keyword arguments passed to the `EfficientDet` constructor.

Returns
-------
- **EfficientDet**:  
  The highest-capacity EfficientDet-D7 variant using EfficientNet-B7 backbone,  
  capable of high-resolution 1536x1536 input detection.

Example Usage
-------------
.. code-block:: python

    >>> from lucid.models import efficientdet_d7
    >>> model = efficientdet_d7(num_classes=80)
    >>> print(model)

    >>> x = lucid.rand(1, 3, 1536, 1536)
    >>> out = model(x)
