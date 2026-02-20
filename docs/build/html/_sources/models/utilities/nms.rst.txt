util.nms
========

.. autofunction:: lucid.models.utils.nms

The `nms` (Non-Maximum Suppression) function filters overlapping 
bounding boxes based on their Intersection over Union (IoU) and confidence scores. 
It is typically used in object detection to remove duplicate predictions.

Function Signature
------------------

.. code-block:: python

    def nms(boxes: Tensor, scores: Tensor, iou_thresh: float = 0.3) -> Tensor

Parameters
----------

- **boxes** (*Tensor*): 
  Tensor of shape :math:`(N, 4)` containing bounding boxes in the format 
  :math:`(x_1, y_1, x_2, y_2)`.
- **scores** (*Tensor*): 
  Tensor of shape :math:`(N,)` representing the confidence score for each box.
- **iou_thresh** (*float*, optional): 
  Threshold for IoU overlap. Boxes with IoU greater than this threshold are suppressed. 
  Defaults to `0.3`.

Returns
-------

- **Tensor**: 
  A 1D integer tensor of indices corresponding to the boxes that are **kept** 
  after suppression.

.. tip::

   The input `boxes` and `scores` must have matching first dimensions. 
   This function is **not differentiable** and is meant for inference or 
   post-processing steps.

Example
-------

.. code-block:: python

    >>> from lucid.models.utils import nms
    >>> boxes = lucid.Tensor([[10, 10, 20, 20], [12, 12, 22, 22], [100, 100, 110, 110]])
    >>> scores = lucid.Tensor([0.9, 0.8, 0.75])
    >>> keep = nms(boxes, scores, iou_thresh=0.5)
    >>> print(keep)
    Tensor([0, 2], dtype=Int64)
