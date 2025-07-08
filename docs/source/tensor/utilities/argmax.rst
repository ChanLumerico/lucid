lucid.argmax
============

.. autofunction:: lucid.argmax

The `argmax` function returns the **indices of the maximum values**
along a specified **axis**.

You can preserve the number of dimensions by setting `keepdims=True`.

Function Signature
------------------

.. code-block:: python

    def argmax(
        a: Tensor,
        axis: int | None = None,
        keepdims: bool = False,
    ) -> Tensor

Parameters
----------

* **a** (*Tensor*):  
  Input tensor to evaluate maximum indices from.

* **axis** (*int* or *None*, optional):  
  Axis along which to find the index of the maximum. If `None`, the input
  is flattened. Defaults to `None`.

* **keepdims** (*bool*, optional):  
  If `True`, retains reduced dimensions with size 1. Defaults to `False`.

Returns
-------

* **Tensor** (*Int64*):  
  Indices of the maximum values along the specified axis.

.. math::

   \operatorname{shape}(\text{out}) \;=\;
   \begin{cases}
   (1, 1, \ldots) & \text{if keepdims=True} \\
   \text{reduced shape} & \text{otherwise}
   \end{cases}

.. note::

   `argmax` is **gradient-free**; back-propagation will not
   propagate through the returned indices.

Examples
--------

.. admonition:: Flattened maximum index

   .. code-block:: python

      >>> x = lucid.Tensor([[3, 2], [5, 4]])
      >>> lucid.argmax(x)
      Tensor(2, grad=None)

.. admonition:: Max index by row

   .. code-block:: python

      >>> lucid.argmax(x, axis=1)
      Tensor([0, 0], grad=None)
