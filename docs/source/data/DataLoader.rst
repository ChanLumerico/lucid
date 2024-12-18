data.DataLoader
===============

.. autoclass:: lucid.data.DataLoader

The `DataLoader` class provides an efficient and flexible way to iterate over datasets in mini-batches.
It supports shuffling, batching, and parallel data loading, making it essential for training deep learning models.

The `DataLoader` works with any dataset that inherits from `Dataset`, enabling seamless integration 
with custom datasets defined by the user.

Class Signature
---------------

.. code-block:: python

    class DataLoader:
        def __init__(
            self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False
        ) -> None

Methods
-------

**Core Methods**

.. code-block:: python

    def __len__(self) -> int

Returns the total number of batches in the `DataLoader`, calculated as the total number of samples
in the dataset divided by the batch size.

**Returns**:

- **int**: The total number of batches.

**Example**:

.. code-block:: python

    dataset = SquareDataset()  # Assume len(dataset) = 10
    loader = DataLoader(dataset, batch_size=2)
    print(len(loader))  # Output: 5

.. code-block:: python

    def __iter__(self) -> Iterator[list[Any]]

Returns an iterator that yields batches of data from the dataset.
The iterator returns a list of samples for each batch.

**Yields**:

- **list[Any]**: A batch of samples, each corresponding to a sample from the dataset.

**Example**:

.. code-block:: python

    dataset = SquareDataset()  # Assume dataset returns squares of indices
    loader = DataLoader(dataset, batch_size=2)
    for batch in loader:
        print(batch)  # Output: [0, 1], [4, 9], ...

**Special Methods**

.. code-block:: python

    def __call__(self) -> Iterator[list[Any]]

An alternative way to obtain an iterator for the `DataLoader`.
This allows for cleaner syntax when using `DataLoader` in loops.

**Yields**:

- **list[Any]**: A batch of samples, each corresponding to a sample from the dataset.

**Example**:

.. code-block:: python

    dataset = SquareDataset()  
    loader = DataLoader(dataset, batch_size=3)
    for batch in loader():  # Call syntax
        print(batch)

Parameters
----------

**__init__**

Initializes the `DataLoader` object with the specified dataset, batch size, and shuffle option.

**Parameters**:

- **dataset** (*Dataset*): The dataset to load data from. Must be a subclass of `Dataset`.
- **batch_size** (*int*, optional): The number of samples per batch. Defaults to 1.
- **shuffle** (*bool*, optional): Whether to shuffle the data at the start of each epoch. 
  Defaults to `False`.

**Raises**:

- **TypeError**: If `dataset` is not an instance of `Dataset`.
- **ValueError**: If `batch_size` is not a positive integer.

Examples
--------

.. admonition:: **Loading a custom dataset**
   :class: note

   .. code-block:: python

       import lucid.data as data

       dataset = data.SquareDataset()
       loader = data.DataLoader(dataset, batch_size=2, shuffle=True)
       
       for batch in loader:
           print(batch)  # Prints random batches of 2 samples from the dataset

.. admonition:: **Using DataLoader with ConcatDataset**
   :class: note

   .. code-block:: python

       import lucid.data as data

       dataset1 = data.SquareDataset()
       dataset2 = data.RandomNoiseDataset(10)
       combined_dataset = data.ConcatDataset([dataset1, dataset2])
       
       loader = data.DataLoader(combined_dataset, batch_size=3, shuffle=True)
       
       for batch in loader:
           print(batch)  # Prints batches of 3 samples from the combined dataset

.. tip:: **Shuffling data**

   When `shuffle=True`, the data order is randomized at the beginning of every epoch.
   This is useful for preventing the model from overfitting to the data order.

   .. code-block:: python

       dataset = SquareDataset()
       loader = DataLoader(dataset, batch_size=2, shuffle=True)
       for epoch in range(2):
           print(f"Epoch {epoch+1}")
           for batch in loader:
               print(batch)  # Batches are randomly shuffled for each epoch

.. warning:: **Batch size constraints**

   If the total number of samples is not a multiple of `batch_size`, 
   the last batch will contain fewer samples than the batch size.

   .. code-block:: python

       dataset = SquareDataset()  # 10 samples in total
       loader = DataLoader(dataset, batch_size=3)
       for batch in loader:
           print(batch)  # Last batch will have fewer than 3 samples
