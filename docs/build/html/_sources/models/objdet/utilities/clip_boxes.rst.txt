util.clip_boxes
===============

.. autofunction:: lucid.models.objdet.util.clip_boxes

The `clip_boxes` function constrains bounding boxes to lie within the spatial 
bounds of the image.

Function Signature
------------------

.. code-block:: python

    def clip_boxes(boxes: Tensor, image_shape: tuple[int, int]) -> Tensor

Parameters
----------

- **boxes** (*Tensor*): 
  Tensor of shape :math:`(N, 4)` representing bounding boxes in the format 
  :math:`(x_1, y_1, x_2, y_2)`.
- **image_shape** (*tuple[int, int]*): 
  Tuple representing the image shape as :math:`(H, W)`.

Returns
-------

- **Tensor**: 
  A new tensor of shape :math:`(N, 4)` where each coordinate is clipped 
  to remain within the image boundaries.

.. note::

   Coordinates are clipped between :math:`[0, W - 1]` for `x` and 
   :math:`[0, H - 1]` for `y`.

Example
-------

.. code-block:: python

    >>> from lucid.models.objdet.util import clip_boxes
    >>> boxes = lucid.Tensor([[10, 10, 120, 130], [-5, -5, 50, 60]])
    >>> clipped = clip_boxes(boxes, image_shape=(100, 100))
    >>> print(clipped)
    Tensor([[10, 10, 99, 99],
            [ 0,  0, 50, 60]], ...)
