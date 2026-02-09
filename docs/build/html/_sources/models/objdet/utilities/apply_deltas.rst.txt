util.apply_deltas
=================

.. autofunction:: lucid.models.utils.apply_deltas

The `apply_deltas` function reconstructs bounding boxes from the regression 
deltas predicted relative to a set of reference boxes. 
It is the inverse of `bbox_to_delta`.

Function Signature
------------------

.. code-block:: python

    def apply_deltas(boxes: Tensor, deltas: Tensor, add_one: float = 1.0) -> Tensor

Parameters
----------

- **boxes** (*Tensor*): 
  Reference boxes of shape :math:`(N, 4)`, with format :math:`(x_1, y_1, x_2, y_2)`.
- **deltas** (*Tensor*): 
  Deltas of shape :math:`(N, 4)` representing predicted changes to the reference boxes.
- **add_one** (*float*, optional): 
  Offset added during width and height computation to prevent zero sizes. Defaults to `1.0`.

Returns
-------

- **Tensor**: 
  Transformed boxes of shape :math:`(N, 4)`, in format :math:`(x_1, y_1, x_2, y_2)`.

.. math::

    \begin{aligned}
    w_s &= x_2 - x_1 + 1 \\
    h_s &= y_2 - y_1 + 1 \\
    x_c &= x_1 + \frac{w_s}{2}, \quad y_c = y_1 + \frac{h_s}{2} \\
    w_t &= \exp(\Delta w) \cdot w_s, \quad h_t = \exp(\Delta h) \cdot h_s \\
    x_c' &= \Delta x \cdot w_s + x_c, \quad y_c' = \Delta y \cdot h_s + y_c \\
    x_1' &= x_c' - \frac{w_t}{2}, \quad x_2' = x_c' + \frac{w_t}{2} - 1 \\
    y_1' &= y_c' - \frac{h_t}{2}, \quad y_2' = y_c' + \frac{h_t}{2} - 1
    \end{aligned}

.. tip::

   This function ensures numerical stability by:
   
   - Clipping the predicted width and height to a minimum value.
   - Swapping coordinates when necessary to maintain :math:`x_1 \leq x_2` 
     and :math:`y_1 \leq y_2`.

Example
-------

.. code-block:: python

    >>> from lucid.models.utils import apply_deltas
    >>> ref_boxes = lucid.Tensor([[10, 10, 20, 20]])
    >>> deltas = lucid.Tensor([[0.1, 0.2, 0.0, 0.1]])
    >>> new_boxes = apply_deltas(ref_boxes, deltas)
    >>> print(new_boxes)
    Tensor([[10.5, 10.9, 20.5, 22.9]], ...)
