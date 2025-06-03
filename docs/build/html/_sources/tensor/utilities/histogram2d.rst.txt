lucid.histogram2d
=================

.. autofunction:: lucid.histogram2d

Function Signature
------------------

.. code-block:: python

    def histogram2d(
        a: Tensor,
        b: Tensor,
        bins: list[int, int] = [10, 10],
        range: list[tuple[int, int]] | None = None,
        density: bool = False,
    ) -> tuple[Tensor, Tensor]

Parameters
----------
- **a** (*Tensor*):
  A 1D tensor representing the a coordinates.

- **b** (*Tensor*):
  A 1D tensor representing the b coordinates.

- **bins** (*tuple[int, int]*, optional):
  Number of bins along a and b. Default is (10, 10).

- **range** (*list of tuples*, optional):
  ((xmin, xmax), (ymin, ymax)) specifying the range of values.

- **density** (*bool*, optional):
  If True, normalize the output histogram.

Returns
-------
- **Tensor**:
  A 2D tensor of bin counts or densities.

- **Tensor**:
  A 2D tensor of bin edges stacked along first axis.

Example
-------

.. code-block:: python

    >>> a = Tensor([0.1, 0.4, 0.3])
    >>> b = Tensor([0.5, 0.2, 0.1])
    >>> hist, edges = histogram2d(a, b, bins=(2, 2), range=((0, 0.5), (0, 0.5)))
