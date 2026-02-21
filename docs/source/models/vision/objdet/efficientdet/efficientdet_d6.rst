efficientdet_d6
===============

.. autofunction:: lucid.models.efficientdet_d6

The `efficientdet_d6` function returns an instance of the `EfficientDet` model,  
with **compound coefficient 6**, optimized for high-accuracy large-scale detection  
with deeper BiFPN and larger input resolution.

**Total Parameters**: 52,634,622

Function Signature
------------------
.. code-block:: python

    def efficientdet_d6(num_classes: int = 80, **kwargs) -> EfficientDet

Parameters
----------
- **num_classes** (*int*, default=80):  
  Number of target object classes to detect.

- **kwargs**:  
  Additional keyword arguments passed to the `EfficientDet` constructor.

Returns
-------
- **EfficientDet**:  
  An instance of the EfficientDet-D6 model using EfficientNet-B6 backbone,  
  trained for 1280x1280 input resolutions.

Example Usage
-------------
.. code-block:: python

    >>> from lucid.models import efficientdet_d6
    >>> model = efficientdet_d6(num_classes=80)
    >>> print(model)

    >>> x = lucid.rand(2, 3, 1280, 1280)
    >>> out = model(x)
