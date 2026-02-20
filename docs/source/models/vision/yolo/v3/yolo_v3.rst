yolo_v3
=======

.. autofunction:: lucid.models.yolo_v3

The `yolo_v3` function returns an instance of the `YOLO_V3` model, 
preconfigured with the original YOLO-v3 architecture and default 3-scale detection.

**Total Parameters**: 62,974,149 (MS-COCO)

Function Signature
------------------
.. code-block:: python

    def yolo_v3(num_classes: int = 80, **kwargs) -> YOLO_V3

Parameters
----------
- **num_classes** (*int*, default=80):  
  Number of object categories to detect. Determines the number of output scores per anchor.

- **kwargs**:  
  Additional keyword arguments passed directly to the `YOLO_V3` constructor.

  Common kwargs include:

  - `anchors` (*list[tuple[int, int]]*, optional): list of 9 anchor boxes in pixel units
  - `image_size` (*int*, default=416): input resolution (typically 416x416)
  - `darknet` (*nn.Module*, optional): custom Darknet-53 style backbone

Returns
-------
- **YOLO_V3**:  
  An initialized instance of the YOLOv3 detection model with 3 detection heads for large,
  medium, and small objects.

Example Usage
-------------
.. code-block:: python

    >>> from lucid.models import yolo_v3
    >>> model = yolo_v3(num_classes=80)
    >>> print(model)

    >>> x = lucid.rand(1, 3, 416, 416)
    >>> out = model(x)
    >>> for o in out:
    ...     print(o.shape)
    # Each output shape: (1, 3 * (5 + 80), H, W)

.. note::

    This model uses 3 detection heads at strides 32, 16, and 8. With an input size of 416,
    these correspond to grid sizes of 13x13, 26x26, and 52x52. Each head predicts 3 bounding
    boxes per cell, with output dimensions based on the number of classes.
