util.iou
========

.. autofunction:: lucid.models.objdet.util.iou

The `iou` function computes the Intersection over Union (IoU) between two sets 
of bounding boxes.

Function Signature
------------------

.. code-block:: python

    def iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor

Parameters
----------

- **boxes_a** (*Tensor*): 
  A tensor of shape :math:`(M, 4)` where each row is a box in the format 
  :math:`(x_1, y_1, x_2, y_2)`.

- **boxes_b** (*Tensor*): 
  A tensor of shape :math:`(N, 4)` representing another set of boxes.

Returns
-------

- **Tensor**: 
  A tensor of shape :math:`(M, N)` where the element at position :math:`(i, j)` 
  contains the IoU between `boxes_a[i]` and `boxes_b[j]`.

.. math::

    \text{IoU}(A, B) = \frac{\text{area}(A \cap B)}{\text{area}(A \cup B)}

.. note::

   - Boxes are expected in absolute coordinate format.
   - This function is **differentiable**, and gradients can be propagated if 
     used in training pipelines.

Example
-------

.. code-block:: python

    >>> from lucid.models.objdet.util import iou
    >>> a = lucid.Tensor([[10, 10, 20, 20]])
    >>> b = lucid.Tensor([[15, 15, 25, 25], [0, 0, 5, 5]])
    >>> iou_matrix = iou(a, b)
    >>> print(iou_matrix)
    Tensor([[0.14285715, 0.        ]], ...)
