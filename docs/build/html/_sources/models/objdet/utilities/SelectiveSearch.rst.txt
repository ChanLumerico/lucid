util.SelectiveSearch
====================

.. autoclass:: lucid.models.objdet.util.SelectiveSearch

The `SelectiveSearch` class implements multi-scale region proposal generation 
using color-based graph segmentation and Non-Maximum Suppression (NMS).

It is typically used in object detection pipelines (e.g., R-CNN) to extract 
object-like candidate boxes prior to classification.

Constructor
-----------

.. code-block:: python

    def __init__(
        scales: tuple[float, ...] = (50, 100, 150, 300),
        min_size: int = 20,
        connectivity: Literal[4, 8] = 8,
        max_boxes: int = 2000,
        iou_thresh: float = 0.8,
    ) -> None

Parameters
----------

- **scales** (*tuple[float, ...]*, optional): 
  A tuple of scale values used to control segmentation granularity. 
  Defaults to `(50, 100, 150, 300)`.
- **min_size** (*int*, optional): 
  Minimum component size in the segmentation. Small components will be merged. 
  Defaults to `20`.
- **connectivity** (*Literal[4, 8]*, optional): 
  Pixel connectivity used during graph segmentation. Defaults to `8`.
- **max_boxes** (*int*, optional): 
  Maximum number of boxes to return per image. Defaults to `2000`.
- **iou_thresh** (*float*, optional): 
  IoU threshold for NMS. Boxes with high overlap are suppressed. Defaults to `0.8`.

Returns
-------

- **SelectiveSearch** (*nn.Module*): 
  A callable module that accepts an image tensor and returns proposed regions.

Example
-------

.. code-block:: python

    >>> from lucid.models.objdet.util import SelectiveSearch
    >>> img = lucid.random.randn(256, 256, 3)
    >>> ss = SelectiveSearch()
    >>> proposals = ss(img)
    >>> print(proposals.shape)
    (N, 4)  # Variable number of proposals
