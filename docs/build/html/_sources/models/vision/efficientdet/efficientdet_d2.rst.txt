efficientdet_d2
===============

.. autofunction:: lucid.models.efficientdet_d2

The `efficientdet_d2` function returns an instance of the `EfficientDet` model,  
configured with **compound coefficient 2**, offering improved accuracy through 
deeper layers and increased input resolution.

**Total Parameters**: 6,457,434

Function Signature
------------------
.. code-block:: python

    def efficientdet_d2(num_classes: int = 80, **kwargs) -> EfficientDet

Parameters
----------
- **num_classes** (*int*, default=80):  
  Number of target object classes to detect.

- **kwargs**:  
  Additional keyword arguments passed to the `EfficientDet` constructor.

Returns
-------
- **EfficientDet**:  
  An instance of the EfficientDet-D2 model with EfficientNet-B2 backbone and 
  scaled BiFPN depth.

Example Usage
-------------
.. code-block:: python

    >>> from lucid.models import efficientdet_d2
    >>> model = efficientdet_d2(num_classes=80)
    >>> print(model)

    >>> x = lucid.rand(2, 3, 768, 768)
    >>> out = model(x)
    >>> for o in out:
    ...     print(o.shape)
