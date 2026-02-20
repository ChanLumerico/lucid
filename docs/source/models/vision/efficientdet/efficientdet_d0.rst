efficientdet_d0
===============

.. autofunction:: lucid.models.efficientdet_d0

The `efficientdet_d0` function returns an instance of the `EfficientDet` model,  
preconfigured with **compound coefficient 0**, corresponding to the smallest and 
most efficient variant of the EfficientDet family.

**Total Parameters**: 3,591,656

Function Signature
------------------
.. code-block:: python

    def efficientdet_d0(num_classes: int = 80, **kwargs) -> EfficientDet

Parameters
----------
- **num_classes** (*int*, default=80):  
  Number of target object classes to detect.

- **kwargs**:  
  Additional keyword arguments passed to the `EfficientDet` constructor.

Returns
-------
- **EfficientDet**:  
  An instance of the EfficientDet-D0 object detector using the EfficientNet-B0 backbone,  
  BiFPN feature fusion, and shared detection heads for classification and regression.

Example Usage
-------------
.. code-block:: python

    >>> from lucid.models import efficientdet_d0
    >>> model = efficientdet_d0(num_classes=80)
    >>> print(model)

    >>> x = lucid.rand(2, 3, 512, 512)
    >>> out = model(x)
    >>> for o in out:
    ...     print(o.shape)
    # Each output tensor corresponds to one image’s detections.
