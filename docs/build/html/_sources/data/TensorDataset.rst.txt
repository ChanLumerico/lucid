data.TensorDataset
==================

.. autoclass:: lucid.data.TensorDataset

Class Signature
---------------

.. code-block:: python

    class TensorDataset(*tensors_or_arrays: Tensor | _ArrayLike)

Parameters
----------
- **tensors_or_arrays** (*Tensor | _ArrayLike*):
  One or more Lucid tensors or array-like objects convertible to Lucid tensors.
  All must have equal length along the first axis. Scalars are not allowed.

Mathematical Shape Constraints
------------------------------

Let :math:`\{\mathbf{T}^{(k)}\}_{k=1}^K` denote the tensors after conversion from
array-likes, each with shape :math:`(N, d_1^{(k)}, \dots, d_{m_k}^{(k)})`. The
length of the dataset is :math:`N` and the sample at index :math:`i` is given by

.. math::

    (\mathbf{T}^{(1)}_{i,:},\; \dots,\; \mathbf{T}^{(K)}_{i,:}).

.. note::

   Array-like inputs are automatically converted to :class:`lucid.Tensor`.

.. warning::

   All tensors must share the same first-dimension size :math:`N`.

Returns
-------
- **__len__** (*int*): Number of samples :math:`N`.
- **__getitem__** (*tuple[Tensor, ...]*): Tuple of indexed tensors for the
  given index `idx`. Supported index types include `int`, `slice`, list,
  tuple, and `lucid.Tensor` (bool/int indexers).

Examples
--------

.. code-block:: python

    import lucid
    from lucid.data import TensorDataset

    x = [[1, 2], [3, 4]]  # array-like
    y = lucid.arange(2)

    ds = TensorDataset(x, y)
    assert len(ds) == 2

    x0, y0 = ds[0]

    # Slice indexing
    xb, yb = ds[:1]

    # Move tensors to GPU (Metal/MLX)
    ds.to("gpu")

Additional Notes
----------------

- `TensorDataset.to(device)` moves all tensors to the specified device
  (`"cpu"` or `"gpu"`).

- Indexing semantics follow :class:`lucid.Tensor`.
- Useful for pairing Lucid tensors or array-likes in supervised learning tasks.
