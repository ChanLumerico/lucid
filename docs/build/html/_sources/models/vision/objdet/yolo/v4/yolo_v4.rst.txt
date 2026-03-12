yolo_v4
=======

.. autofunction:: lucid.models.yolo_v4

The `yolo_v4` function returns an instance of the `YOLO_V4` model,  
preconfigured with the YOLO-v4 architecture and default 3-scale detection pipeline.

**Total Parameters**: 93,488,078 (Full CSP-Backbone)

Function Signature
------------------
.. code-block:: python

    def yolo_v4(num_classes: int = 80, **kwargs) -> YOLO_V4

Parameters
----------
- **num_classes** (*int*, default=80):  
  Number of object categories to detect. Determines number of logits per anchor.

- **kwargs**:  
  Additional keyword arguments passed to `YOLO_V4Config`. By default this
  factory uses the standard YOLO-v4 anchors/strides and disables the IoU-aware
  branch weighting path with `iou_aware_alpha=0.0` and `iou_branch_weight=0.0`.

Returns
-------
- **YOLO_V4**:  
  An instance of the YOLOv4 object detector, featuring SPP and PAN necks, 
  with 3 detection heads.

Example Usage
-------------
.. code-block:: python

    >>> import lucid
    >>> from lucid.models import yolo_v4
    >>> model = yolo_v4(num_classes=80)
    >>> print(model)

    >>> x = lucid.ones(1, 3, 416, 416)
    >>> out = model(x)
    >>> for o in out:
    ...     print(o.shape)
    # Each output shape: (1, 3 * (5 + 80), H, W)

.. note::

    YOLO-v4 improves on YOLO-v3 by incorporating:
    
    - **CSPDarknet-53** backbone (faster & more efficient),
    - **SPP** (Spatial Pyramid Pooling) for receptive field expansion,
    - **PANet** for enhanced bottom-up feature aggregation,
    - **DropBlock**, **Mish activation**, and **IoU-aware objectness** f
      or improved performance.
