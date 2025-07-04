lucid.argmin
============

.. autofunction:: lucid.argmin

The `argmin` function returns the **indices of the minimum values**
along a specified **axis**.

You can preserve the number of dimensions by setting `keepdims=True`.

Function Signature
------------------

.. code-block:: python

    def argmin(
        a: Tensor,
        axis: int | None = None,
        keepdims: bool = False,
    ) -> Tensor

Parameters
----------

* **a** (*Tensor*):  
  Input tensor to evaluate minimum indices from.

* **axis** (*int* or *None*, optional):  
  Axis along which to find the index of the minimum. If `None`, the input
  is flattened. Defaults to `None`.

* **keepdims** (*bool*, optional):  
  If `True`, retains reduced dimensions with size 1. Defaults to `False`.

Returns
-------

* **Tensor** (*Int64*):  
  Indices of the minimum values along the specified axis.

.. math::

   \operatorname{shape}(\text{out}) \;=\;
   \begin{cases}
   (1, 1, \ldots) & \text{if keepdims=True} \\
   \text{reduced shape} & \text{otherwise}
   \end{cases}

.. note::

   `argmin` is **gradient-free**; back-propagation will not
   propagate through the returned indices.

Examples
--------

.. admonition:: Global minimum index

   .. code-block:: python

      >>> x = lucid.Tensor([[3, 2], [1, 4]])
      >>> lucid.argmin(x)
      Tensor(2, grad=None)

.. admonition:: Along axis, keeping dims

   .. code-block:: python

      >>> lucid.argmin(x, axis=1, keepdims=True)
      Tensor([[1], [0]], grad=None)
