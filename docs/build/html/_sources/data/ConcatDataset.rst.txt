lucid.data.ConcatDataset
========================

.. autoclass:: lucid.data.ConcatDataset

The `ConcatDataset` class enables the combination of multiple datasets into a single unified dataset.
This is useful when you have separate datasets that you want to treat as one large dataset.

By providing a list of datasets, `ConcatDataset` allows seamless iteration through 
all the samples as if they belonged to a single dataset.

Class Signature
---------------

.. code-block:: python

    class ConcatDataset(Dataset):
        def __init__(self, datasets: list[Dataset]) -> None

Methods
-------

**Core Methods**

.. code-block:: python

    def __len__(self) -> int

Returns the total number of samples across all concatenated datasets.

**Returns**:

- **int**: The total number of samples from all combined datasets.

**Example**:

.. code-block:: python

    dataset1 = SquareDataset()  # Assume len(dataset1) = 10
    dataset2 = SquareDataset()  # Assume len(dataset2) = 10
    combined_dataset = ConcatDataset([dataset1, dataset2])
    print(len(combined_dataset))  # Output: 20

**Raises**:

- **TypeError**: If any of the elements in the `datasets` list is not an instance of `Dataset`.

.. code-block:: python

    def __getitem__(self, index: int) -> Any

Fetches a sample from the concatenated datasets.

The method determines which underlying dataset to access using the 
index and retrieves the corresponding sample.

**Parameters**:

- **index** (*int*): The global index of the sample to retrieve.

**Returns**:

- **Any**: The data sample corresponding to the specified index.

**Example**:

.. code-block:: python

    dataset1 = SquareDataset()  # Assume dataset1 has samples [0, 1, 4, 9, ...]
    dataset2 = SquareDataset()  # Assume dataset2 has samples [0, 1, 4, 9, ...]
    combined_dataset = ConcatDataset([dataset1, dataset2])
    print(combined_dataset[0])  # Output from dataset1, e.g., 0
    print(combined_dataset[15])  # Output from dataset2, e.g., 25

**Raises**:

- **IndexError**: If the index is out of range for the concatenated dataset.

**Special Methods**

.. code-block:: python

    def __iter__(self) -> Iterator[Any]

Returns an iterator that iterates over all samples in the concatenated datasets.

**Yields**:

- **Any**: The samples of the concatenated datasets, one at a time.

Examples
--------

.. admonition:: **Combining multiple datasets**
   :class: note

   .. code-block:: python

       import lucid.data as data

       dataset1 = data.SquareDataset()
       dataset2 = data.RandomNoiseDataset(10)
       combined_dataset = data.ConcatDataset([dataset1, dataset2])
       
       print(len(combined_dataset))  # Total number of samples = len(dataset1) + len(dataset2)
       
       for sample in combined_dataset:
           print(sample)  # Iterates through samples from dataset1 followed by dataset2

.. tip:: **Dynamic dataset concatenation**

   You can dynamically concatenate datasets with different sizes or structures,
   but be mindful that the `__getitem__` method must handle the index calculation properly.

.. warning:: **Index errors**

   Be cautious when accessing samples from large concatenated datasets.
   If the provided index is outside the range of combined dataset indices, 
   an **IndexError** will be raised.

   .. code-block:: python

       combined_dataset = ConcatDataset([SquareDataset(), SquareDataset()])
       
       try:
           sample = combined_dataset[999]  # Assuming the combined dataset has less than 999 samples
       except IndexError as e:
           print("Index out of range!", e)
