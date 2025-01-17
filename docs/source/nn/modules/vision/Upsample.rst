nn.Upsample
===========

.. autoclass:: lucid.nn.Upsample

The `Upsample` module performs upsampling on the input tensor using specified interpolation methods.
It allows flexibility in defining the output size either directly or by specifying a scaling factor.

Class Signature
---------------

.. code-block:: python

    class lucid.nn.Upsample(
        size: tuple[int, int] | None = None,
        scale_factor: float | tuple[float, float] | None = None,
        mode: _InterpolateType = "nearest",
        align_corners: bool | None = None,
    )

Parameters
----------
- **size** (*tuple[int, int] | None, optional*):
  The desired output size (height, width). If None, 
  the size is calculated using `scale_factor`.

- **scale_factor** (*float | tuple[float, float] | None, optional*):
  The multiplier for spatial dimensions. Used to calculate the output size when 
  `size` is None. Default is None.

- **mode** (*str, optional*):
  The interpolation method to use. Supported values include `bilinear`, `nearest`, etc. 
  Default is `nearest`.

- **align_corners** (*bool | None, optional*):
  If True, aligns the corner pixels of the input and output tensors. 
  Has no effect when `mode` is `nearest`. Default is None.

Examples
--------

**Upsampling using size:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[1, 2], [3, 4]], requires_grad=True).reshape(1, 1, 2, 2)
    >>> upsample = nn.Upsample(size=(4, 4), mode='bilinear', align_corners=True)
    >>> output = upsample(input_tensor)
    >>> print(output)

**Upsampling using scale_factor:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[1, 2], [3, 4]], requires_grad=True).reshape(1, 1, 2, 2)
    >>> upsample = nn.Upsample(scale_factor=2.0, mode='nearest')
    >>> output = upsample(input_tensor)
    >>> print(output)

.. important::

  - At least one of `size` or `scale_factor` must be specified during initialization.
  - If both `size` and `scale_factor` are provided, `size` takes precedence.
