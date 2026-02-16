util.bbox_to_delta
==================

.. autofunction:: lucid.models.utils.bbox_to_delta

The `bbox_to_delta` function computes the offset deltas (translation and scale) 
required to transform a set of source boxes into target boxes. This is commonly 
used in object detection for bounding box regression.

Function Signature
------------------

.. code-block:: python

    def bbox_to_delta(src: Tensor, target: Tensor, add_one: float = 1.0) -> Tensor

Parameters
----------

- **src** (*Tensor*): 
  Source bounding boxes of shape :math:`(N, 4)` in format :math:`(x_1, y_1, x_2, y_2)`.
- **target** (*Tensor*): 
  Target bounding boxes of shape :math:`(N, 4)` in the same format as `src`.
- **add_one** (*float*, optional): 
  Offset to avoid log(0) or zero width/height when computing scale. Defaults to `1.0`.

Returns
-------

- **Tensor**: A tensor of shape :math:`(N, 4)` containing the deltas in the form:

  .. math::

      \begin{aligned}
      \Delta x &= \frac{x_t - x_s}{w_s} \\
      \Delta y &= \frac{y_t - y_s}{h_s} \\
      \Delta w &= \log\left(\frac{w_t}{w_s + \epsilon}\right) \\
      \Delta h &= \log\left(\frac{h_t}{h_s + \epsilon}\right)
      \end{aligned}

Where :math:`(x_s, y_s, w_s, h_s)` and :math:`(x_t, y_t, w_t, h_t)` are the center 
coordinates and sizes of `src` and `target`, respectively.

Example
-------

.. code-block:: python

    >>> from lucid.models.utils import bbox_to_delta
    >>> src = lucid.Tensor([[10, 10, 20, 20]])
    >>> tgt = lucid.Tensor([[12, 12, 24, 28]])
    >>> delta = bbox_to_delta(src, tgt)
    >>> print(delta)
    Tensor([[0.2, 0.2, 0.182, 0.336]], ...)
