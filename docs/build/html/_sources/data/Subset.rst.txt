data.Subset
===========

.. autoclass:: lucid.data.Subset

The `Subset` class provides a view over a portion of another dataset.  
It behaves like the original dataset but only exposes a subset of its samples,  
allowing partial training, validation, or testing splits while maintaining  
the same interface and behavior as the original dataset.

Class Signature
---------------

.. code-block:: python

    class Subset(dataset: Dataset, indices: list[int])

Parameters
----------

- **dataset** (:class:`lucid.data.Dataset`):
  The original dataset from which this subset is derived.  
  The subset delegates all method and attribute access to this dataset.

- **indices** (*list[int]*):
  The list of indices representing which elements of the original dataset  
  belong to this subset.

Attributes
----------

- **dataset** (:class:`lucid.data.Dataset`):
  The original dataset instance that this subset wraps.

- **indices** (*list[int]*):
  The subset indices that determine which samples are included.

Methods
-------

- **__getitem__(index: int) -> Any**:
  Returns the item at the given position within the subset.

- **__len__() -> int**:
  Returns the number of samples in the subset.

- **__iter__() -> Iterator[Any]**:
  Returns an iterator over the subset samples.

- **__getattr__(name: str) -> Any**:
  Forwards attribute access to the underlying dataset.  
  This allows the subset to mirror the methods and properties of its parent dataset.

Examples
--------

.. code-block:: python

    >>> from lucid.data import Dataset, Subset

    >>> class MyDataset(Dataset):
    ...     def __init__(self):
    ...         self.data = list(range(10))
    ...     def __getitem__(self, idx):
    ...         return self.data[idx]
    ...     def __len__(self):
    ...         return len(self.data)

    >>> full_dataset = MyDataset()
    >>> subset = Subset(full_dataset, [0, 1, 2, 3])
    >>> len(subset)
    4
    >>> list(subset)
    [0, 1, 2, 3]

.. note::

   `Subset` preserves the full functionality of the original dataset,  
   including any custom attributes, transforms, or preprocessing methods.
