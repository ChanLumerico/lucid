lucid.expand
============

.. autofunction:: lucid.expand

The `expand` function returns a view of a tensor with singleton dimensions expanded
(or new leading dimensions added) without allocating new data.

Function Signature
------------------

.. code-block:: python

    def expand(a: Tensor, *sizes: int | _ShapeLike) -> Tensor

Parameters
----------

- **a** (*Tensor*):
  The input tensor to expand.

- **sizes** (*int | _ShapeLike*):
  Target shape. You can pass a tuple/list or variadic integers. Use `-1`
  to keep the corresponding input dimension size unchanged.

Returns
-------

- **Tensor**:
  A view of the input with expanded shape. If `a` requires gradients, the
  returned tensor will also track gradients.

Rules
-----

- If the target has more dimensions than the input, leading dimensions are
  prepended with size `1`.

- A dimension can be expanded only if the input dimension is `1` or matches
  the target size.

- `-1` is allowed only for existing (non-leading) dimensions and means
  "keep the input size".

- Incompatible shapes raise an error.

.. note::

   This function does not copy data; it creates a broadcasted view.

Examples
--------

**Basic Example**

.. code-block:: python

    >>> import lucid
    >>> a = lucid.Tensor([[1], [2], [3]])  # Shape: (3, 1)
    >>> b = lucid.expand(a, -1, 4)
    >>> b.shape
    (3, 4)

**Add Leading Dimensions**

.. code-block:: python

    >>> a = lucid.Tensor([1, 2, 3])  # Shape: (3,)
    >>> b = lucid.expand(a, 2, 3)
    >>> b.shape
    (2, 3)

Backward Gradient Calculation
-----------------------------

Gradients are summed along expanded dimensions (the same rule as broadcasting).

.. math::

    \frac{\partial L}{\partial a} = \sum_{\text{expanded axes}} \frac{\partial L}{\partial b}

**Example with Gradient Computation**

.. code-block:: python

    >>> a = lucid.arange(3, requires_grad=True)
    >>> b = a.reshape(-1, 1)
    >>> c = lucid.expand(b, -1, 4)
    >>> d = c[:, ::2].sum()
    >>> d.backward()
    >>> a.grad
    array([2., 2., 2.])
