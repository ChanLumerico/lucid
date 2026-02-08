lucid.histogram
===============

.. autofunction:: lucid.histogram

Function Signature
------------------

.. code-block:: python

    def histogram(
        a: Tensor,
        bins: int = 10,
        range: tuple[float, float] | None = None,
        density: bool = False,
    ) -> tuple[Tensor, Tensor]

Parameters
----------
- **a** (*Tensor*):
  A 1D tensor of values to bin.

- **bins** (*int*, optional):
  Number of histogram bins. Default is 10.

- **range** (*tuple[float, float]*, optional):
  Minimum and maximum range of the histogram. If None, uses data min/max.

- **density** (*bool*, optional):
  If True, the result is normalized to form a probability density.

Returns
-------
- **Tensor**:
  A 1D tensor of bin counts or densities.

- **Tensor**:
  A 1D tensor of bin edges.

Example
-------

.. code-block:: python

    >>> x = Tensor([0.1, 0.2, 0.3, 0.4])
    >>> hist, edges = histogram(x, bins=2, range=(0.0, 0.5))