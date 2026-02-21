efficientdet_d1
===============

.. autofunction:: lucid.models.efficientdet_d1

The `efficientdet_d1` function returns an instance of the `EfficientDet` model,  
preconfigured with **compound coefficient 1**, representing a deeper and 
higher-resolution variant compared to EfficientDet-D0.

**Total Parameters**: 5,068,752

Function Signature
------------------
.. code-block:: python

    def efficientdet_d1(num_classes: int = 80, **kwargs) -> EfficientDet

Parameters
----------
- **num_classes** (*int*, default=80):  
  Number of target object classes to detect.

- **kwargs**:  
  Additional keyword arguments passed to the `EfficientDet` constructor.

Returns
-------
- **EfficientDet**:  
  An instance of the EfficientDet-D1 detector using the EfficientNet-B1 backbone,  
  with a larger BiFPN and higher input resolution than D0.

Example Usage
-------------
.. code-block:: python

    >>> from lucid.models import efficientdet_d1
    >>> model = efficientdet_d1(num_classes=80)
    >>> print(model)

    >>> x = lucid.rand(2, 3, 640, 640)
    >>> out = model(x)
    >>> for o in out:
    ...     print(o.shape)
    # Each output tensor corresponds to one image’s detections.
