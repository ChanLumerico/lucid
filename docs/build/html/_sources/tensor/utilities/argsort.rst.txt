lucid.argsort
=============

.. autofunction:: lucid.argsort

The `argsort` function returns the indices that would sort a tensor along a
specified **axis**.  

It supports **ascending / descending order**, lets you choose the backend
`kind` (quicksort, mergesort, heapsort, or stable), and offers an
additional **stable** flag(only for CPU) for convenience.

Function Signature
------------------

.. code-block:: python

    def argsort(
        a: Tensor,
        axis: int = -1,
        descending: bool = False,
        kind: _SortKind | None = None,
        stable: bool = False,
    ) -> Tensor

Parameters
----------

* **a** (*Tensor*):  
  Input tensor whose elements will be ordered.

* **axis** (*int*, optional):  
  Axis along which to sort. Negative values count from the end.
  Defaults to `-1` (the last axis).

* **descending** (*bool*, optional):  
  If `True` the indices correspond to **descending** order.
  Defaults to `False` (ascending).

* **kind** (*{ "quicksort" | "mergesort" | "heapsort" | "stable" }* or *None*):  
  Sorting algorithm to use (passed straight to selected backends).  
  If *None*, Lucid picks `"quicksort"` unless `stable=True`, in which
  case it chooses `"stable"` automatically.

* **stable** (*bool*, optional):  
  Ensures that the relative order of equal elements is preserved
  (**stable sort**).  Overrides *kind* to `"stable"` when set.  
  Supported on CPU; on GPU the flag is accepted but currently ignored
  with a warning.  Defaults to `False`.

Returns
-------

* **Tensor** (*Int32*):
  A tensor of indices with the **same shape** as *a*; indexing *a* with this
  tensor along *axis* yields a sorted view of the original data.

.. math::

   \operatorname{shape}(\text{out}) \;=\; \operatorname{shape}(a)

.. note::

   `argsort` is **gradient-free**; the output indices never track
   gradients, and back-propagation does **not** modify `a.grad`.

Examples
--------

.. admonition:: 1-D ascending

   .. code-block:: python

      >>> import lucid
      >>> x = lucid.Tensor([3, 1, 4, 1])
      >>> idx = lucid.argsort(x)
      >>> x[idx]          # sorted view
      Tensor([1, 1, 3, 4], grad=None)

.. admonition:: Descending, heapsort kernel

   .. code-block:: python

      >>> lucid.argsort(x, descending=True, kind="heapsort")
      Tensor([2, 0, 1, 3], grad=None)

.. admonition:: Stable sort with duplicates

   .. code-block:: python

      >>> z = lucid.Tensor([10, 10, 10, 9])
      >>> lucid.argsort(z, stable=True)
      Tensor([3, 0, 1, 2], grad=None)   # duplicates keep insertion order

Performance / FLOPs
-------------------

For each slice of length *n* along *axis*, the complexity is 

.. math::

   \mathcal{O}\!\left(n \log n\right)

(with a smaller constant factor when MLX chooses radix-sort on GPU).
