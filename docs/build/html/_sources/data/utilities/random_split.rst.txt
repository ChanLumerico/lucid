data.random_split
=================

.. autofunction:: lucid.data.random_split

The `random_split` function divides a dataset into non-overlapping subsets  
of specified lengths or fractional proportions. This function is often used  
to create train/validation/test splits in data pipelines.

Function Signature
------------------

.. code-block:: python

    def random_split(
        dataset: Dataset,
        lengths: Sequence[int | float],
        seed: int | None = None
    ) -> tuple[Subset, ...]

Parameters
----------

- **dataset** (:class:`lucid.data.Dataset`):
  The dataset to split. Must implement `__len__` and `__getitem__`.

- **lengths** (*Sequence[int | float]*):
  The lengths (or fractions) of each split.
  
  - If integers, they must sum to `len(dataset)`.
  - If floats, they are treated as fractions that must sum to 1.0.

- **seed** (*int*, optional):
  A random seed for reproducibility.  
  If `None`, it uses global random seed from `lucid.random` package.

Returns
-------

- **tuple[Subset, ...]**:
  A tuple of :class:`lucid.data.Subset` instances,  
  each wrapping the original dataset and exposing only its respective samples.

Mathematical Explanation
------------------------

.. math::

    \text{indices} = \text{Shuffle}(\{0, 1, ..., N-1\})

    \text{subset}_k = \{ \text{dataset}[i] \mid i \in \text{indices}_{k} \}

where:

- :math:`N` is the total number of samples in the dataset.
- :math:`\text{indices}_k` represents the segment of indices assigned to the k-th subset.

Examples
--------

**Splitting by integer lengths:**

.. code-block:: python

    >>> from lucid.data import random_split
    >>> data = list(range(10))
    >>> train_ds, val_ds = random_split(data, [8, 2], seed=42)
    >>> len(train_ds), len(val_ds)
    (8, 2)

**Splitting by fractional proportions:**

.. code-block:: python

    >>> from lucid.data import random_split
    >>> data = list(range(10))
    >>> train_ds, val_ds = random_split(data, [0.8, 0.2], seed=1)
    >>> len(train_ds), len(val_ds)
    (8, 2)

.. tip::

   You can safely use each split with :class:`lucid.data.DataLoader`  
   without additional wrapping or conversions.
