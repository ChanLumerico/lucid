yolo_v2_tiny
============

.. autofunction:: lucid.models.yolo_v2_tiny

The `yolo_v2_tiny` function returns an instance of the `YOLO_V2` model,
configured with a lightweight version of the original YOLO-v2 architecture using 
a compact DarkNet-19 variant.

**Total Parameters**: 15,863,821

Function Signature
------------------
.. code-block:: python

    def yolo_v2_tiny(num_classes: int = 20, **kwargs) -> YOLO_V2

Parameters
----------
- **num_classes** (*int*, default=20):  
  Number of object categories to detect. This sets the number of output class scores per 
  anchor in the final detection head.

- **kwargs**:  
  Additional keyword arguments passed directly to the `YOLO_V2` constructor.

  Common kwargs include:

  - `darknet` (*nn.Module*, optional): custom lightweight backbone
  - `num_anchors` (*int*, default=5): number of anchor boxes (defaults to 5)
  - `image_size` (*int*, default=416): input image resolution

Returns
-------
- **YOLO_V2**:
  An initialized instance of the `YOLO_V2` detection model using a tiny backbone.

Example Usage
-------------
.. code-block:: python

    >>> from lucid.models import yolo_v2_tiny
    >>> model = yolo_v2_tiny(num_classes=20)
    >>> print(model)

    >>> x = lucid.rand(1, 3, 416, 416)
    >>> out = model(x)
    >>> print(out.shape)  # shape: (1, 125, 13, 13) for Pascal-VOC (20 classes, 5 anchors)

.. note::

    The tiny version uses a compressed version of the DarkNet-19 backbone, 
    significantly reducing model size and computation while maintaining similar output 
    format to the original YOLO-v2.

    The model follows the YOLO-v2 head design with 5 anchor boxes and a grid output shape of 
    :math:`S \times S \times (B \times (5 + C))`, where :math:`C` is the number of classes, 
    :math:`B=5` is the number of anchors, and :math:`S=13` when input image size is 416.
