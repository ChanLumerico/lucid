efficientdet_d5
===============

.. autofunction:: lucid.models.efficientdet_d5

The `efficientdet_d5` function returns an instance of the `EfficientDet` model,  
configured with **compound coefficient 5**, scaling both feature and input resolutions  
for enhanced detection of small and medium-sized objects.

**Total Parameters**: 29,882,556

Function Signature
------------------
.. code-block:: python

    def efficientdet_d5(num_classes: int = 80, **kwargs) -> EfficientDet

Parameters
----------
- **num_classes** (*int*, default=80):  
  Number of target object classes to detect.

- **kwargs**:  
  Additional keyword arguments passed to the `EfficientDet` constructor.

Returns
-------
- **EfficientDet**:  
  An instance of EfficientDet-D5 using EfficientNet-B5 backbone and 1280x1280 
  input resolution.

Example Usage
-------------
.. code-block:: python

    >>> from lucid.models import efficientdet_d5
    >>> model = efficientdet_d5(num_classes=80)
    >>> print(model)

    >>> x = lucid.rand(2, 3, 1280, 1280)
    >>> out = model(x)
