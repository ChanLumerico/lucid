data.Dataset
============

.. autoclass:: lucid.data.Dataset

The `Dataset` class is a core abstraction within the `lucid.data` package.
It provides a consistent interface for defining datasets, enabling easy 
integration with `DataLoader` to streamline the training process.

Users subclass `Dataset` to create custom datasets that define how data samples are accessed, 
processed, and returned. By implementing the `__len__` and `__getitem__` methods, users can 
efficiently work with datasets of any size and complexity.

Class Signature
---------------

.. code-block:: python

    class Dataset(ABC):
        def __init__(self) -> None

Methods
-------

**Core Methods**

.. code-block:: python

    def __len__(self) -> int

Returns the total number of samples in the dataset.

**Returns**:

- **int**: The total number of samples in the dataset.

**Raises**:

- **NotImplementedError**: If not implemented by the subclass.

.. code-block:: python

    def __getitem__(self, index: int) -> Any

Retrieves a sample from the dataset at the specified index.

**Parameters**:

- **index** (*int*): The index of the data sample to retrieve.

**Returns**:

- **Any**: The sample corresponding to the specified index.
  This can be a tensor, a tuple, or any other custom format required for training.

**Raises**:

- **NotImplementedError**: If not implemented by the subclass.

Examples
--------

.. admonition:: **Defining a custom dataset**
   :class: note

   .. code-block:: python

       import lucid.data as data

       class SquareDataset(data.Dataset):
           def __len__(self):
               return 10
           
           def __getitem__(self, idx):
               return idx ** 2

       dataset = SquareDataset()
       print(len(dataset))  # Output: 10
       print(dataset[2])  # Output: 4

.. tip:: **Using the Dataset with DataLoader**

   Use the `DataLoader` to iterate over batches of samples from the `Dataset`.

   .. code-block:: python

       from lucid.data import DataLoader
       
       dataset = SquareDataset()
       loader = DataLoader(dataset, batch_size=3, shuffle=True)
       
       for batch in loader:
           print(batch)  # Prints batches of 3 samples

.. warning:: **Incomplete implementation of abstract methods**

   If `__len__` or `__getitem__` are not implemented in a subclass, 
   instantiating the subclass will raise a **TypeError**.

   .. code-block:: python

       class IncompleteDataset(data.Dataset):
           pass
       
       # This will raise a TypeError because __len__ and __getitem__ are not defined.
       dataset = IncompleteDataset()

.. attention:: **Dynamic Datasets**

   For datasets with on-the-fly transformations, override `__getitem__` to apply the transformation
   dynamically for each sample. This avoids the need to store transformed data, saving memory.

   .. code-block:: python

       import random
       
       class RandomNoiseDataset(data.Dataset):
           def __init__(self, size):
               self.size = size
           
           def __len__(self):
               return self.size
           
           def __getitem__(self, idx):
               return random.random()  # Return a random float
       
       dataset = RandomNoiseDataset(5)
       for sample in dataset:
           print(sample)  # Prints 5 random numbers

