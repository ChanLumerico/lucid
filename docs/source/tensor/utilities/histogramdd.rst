lucid.histogramdd
=================

.. autofunction:: lucid.histogramdd

Function Signature
------------------

.. code-block:: python

    def histogramdd(
        a: Tensor,
        /,
        bins: list[int],
        range: list[tuple[float, float]],
        density: bool = False,
    ) -> tuple[Tensor, Tensor]

Parameters
----------
- **a** (*Tensor*):
  A 2D tensor of shape (N, D) representing N D-dimensional samples.

- **bins** (*list[int]*):
  Number of bins for each dimension.

- **range** (*list[tuple[float, float]]*):
  Lower and upper range of each dimension.

- **density** (*bool*, optional):
  If True, the result is normalized to form a probability density.

Returns
-------
- **Tensor**:
  An N-dimensional tensor representing the histogram.

- **Tensor**:
  A 2D tensor of shape (D, B+1) containing the bin edges for each dimension.

Example
-------

.. code-block:: python

    >>> x = Tensor([[0.5, -0.2], [1.0, 0.3], [-0.8, 0.7]])
    >>> hist, edges = histogramdd(x, bins=[4, 4], range=[(-1, 1), (-1, 1)])