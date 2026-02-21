efficientdet_d4
===============

.. autofunction:: lucid.models.efficientdet_d4

The `efficientdet_d4` function returns an instance of the `EfficientDet` model,  
configured with **compound coefficient 4**, featuring larger backbone, deeper BiFPN,  
and higher input resolution for more accurate object detection.

**Total Parameters**: 18,740,232

Function Signature
------------------
.. code-block:: python

    def efficientdet_d4(num_classes: int = 80, **kwargs) -> EfficientDet

Parameters
----------
- **num_classes** (*int*, default=80):  
  Number of target object classes to detect.

- **kwargs**:  
  Additional keyword arguments passed to the `EfficientDet` constructor.

Returns
-------
- **EfficientDet**:  
  An instance of the EfficientDet-D4 model with EfficientNet-B4 backbone.

Example Usage
-------------
.. code-block:: python

    >>> from lucid.models import efficientdet_d4
    >>> model = efficientdet_d4(num_classes=80)
    >>> print(model)

    >>> x = lucid.rand(2, 3, 1024, 1024)
    >>> out = model(x)
