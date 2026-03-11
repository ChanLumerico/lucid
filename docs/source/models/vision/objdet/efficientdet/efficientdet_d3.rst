efficientdet_d3
===============

.. autofunction:: lucid.models.efficientdet_d3

The `efficientdet_d3` function returns an instance of the `EfficientDet` model,  
preconfigured with **compound coefficient 3**, providing balanced performance between  
accuracy and efficiency for general-purpose detection tasks.

**Total Parameters**: 10,286,134

Function Signature
------------------
.. code-block:: python

    def efficientdet_d3(num_classes: int = 80, **kwargs) -> EfficientDet

Parameters
----------
- **num_classes** (*int*, default=80):  
  Number of target object classes to detect.

- **kwargs**:  
  Additional keyword arguments passed to `EfficientDetConfig`. This factory fixes the variant-specific `compound_coef` preset.

Returns
-------
- **EfficientDet**:  
  An instance of the EfficientDet-D3 detector built with EfficientNet-B3 backbone  
  and multi-level BiFPN.

Example Usage
-------------
.. code-block:: python

    >>> import lucid
    >>> from lucid.models import efficientdet_d3
    >>> model = efficientdet_d3(num_classes=80)
    >>> print(model)

    >>> x = lucid.ones(2, 3, 896, 896)
    >>> out = model(x)
