yolo_v3_tiny
============

.. autofunction:: lucid.models.yolo_v3_tiny

The `yolo_v3_tiny` function returns an instance of the `YOLO_V3` model configured
with a lightweight backbone similar to the official YOLOv3-Tiny design.

**Total Parameters**: 23,450,997 (MS-COCO)

Function Signature
------------------
.. code-block:: python

    def yolo_v3_tiny(num_classes: int = 80, **kwargs) -> YOLO_V3

Parameters
----------
- **num_classes** (*int*, default=80):  
  Number of object categories to detect. This controls the number of class predictions 
  per anchor.

- **kwargs**:  
  Additional keyword arguments passed to `YOLO_V3Config`. By default this
  factory uses the tiny 3-scale backbone with
  `darknet_out_channels_arr=[128, 256, 512]` and `image_size=416`.

Returns
-------
- **YOLO_V3**:  
  A YOLOv3 model instance with reduced complexity and parameter count,
  optimized for real-time applications and resource-limited environments.

Example Usage
-------------
.. code-block:: python

    >>> import lucid
    >>> from lucid.models import yolo_v3_tiny
    >>> model = yolo_v3_tiny(num_classes=80)
    >>> print(model)

    >>> x = lucid.ones(1, 3, 416, 416)
    >>> out = model(x)
    >>> for o in out:
    ...     print(o.shape)

.. note::

    This Tiny variant uses the same 3-scale detection architecture as the full YOLOv3 model,
    but with a more efficient backbone. Each detection head processes feature maps of sizes
    13x13, 26x26, and 52x52 (for 416x416 input), using 3 anchors per scale.
