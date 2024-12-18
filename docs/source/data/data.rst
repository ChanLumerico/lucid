lucid.data
==========

The `lucid.data` package provides essential tools for handling datasets and data loading, 
making it easier to prepare and feed data into deep learning models. 

It includes core components for defining datasets and iterating over them efficiently.

Overview
--------

The `lucid.data` package is designed to streamline the data pipeline for machine learning 
models by providing:

- **Dataset Abstraction**: A flexible `Dataset` class for defining custom datasets.
  
- **Data Loading**: The `DataLoader` utility for efficient and parallel data iteration.
  
- **Batching and Shuffling**: Built-in support for batch processing, data shuffling, 
  and other essential functionalities for training.

Key Components
--------------

.. rubric:: `Dataset`

The `Dataset` class serves as an abstract base class for all datasets. It allows users to 
define how data is accessed and processed. To create a custom dataset, subclass `Dataset` 
and implement the `__len__` and `__getitem__` methods.

.. important::

    The `Dataset` class defines how data samples are accessed. Each sample can be a tensor, 
    a tuple, or any other custom format required for training. Users must override `__len__` 
    to define the dataset size and `__getitem__` to specify how a sample is accessed.

.. admonition:: Example

    Here’s a simple example of a custom dataset that returns squares of integers from 0 to 9:

    .. code-block:: python

        >>> import lucid
        >>> import lucid.data as data
        >>>
        >>> class SquareDataset(data.Dataset):
        ...     def __len__(self):
        ...         return 10
        ...     
        ...     def __getitem__(self, idx):
        ...         return idx ** 2
        
        >>> dataset = SquareDataset()
        >>> print(len(dataset))
        10
        >>> print(dataset[2])
        4
    
    In this example, `dataset` provides access to squared integers from 0 to 9.

.. caution::

    When working with large datasets, ensure `__getitem__` is optimized for efficiency.
    Computationally expensive operations inside `__getitem__` may slow down data loading, 
    especially if not used with a `DataLoader`.

.. rubric:: `DataLoader`

The `DataLoader` provides an iterable over a `Dataset`, supporting batching, shuffling, 
and parallel data loading. It is essential for training models with large datasets, as it 
efficiently handles the loading of data in batches.

.. tip::

    Use `DataLoader` to iterate over datasets efficiently, especially when training models.
    It enables batching, shuffling, and parallel data loading to maximize performance.

.. admonition:: Example

    Here’s an example of how to use `DataLoader` to batch data from a custom dataset:

    .. code-block:: python

        >>> dataset = SquareDataset()
        >>> loader = data.DataLoader(dataset, batch_size=3, shuffle=True)
        >>> for batch in loader:
        ...     print(batch)
    
    In this example, `DataLoader` returns batches of 3 samples at a time.
    The `shuffle=True` argument ensures the order of the data is randomized in each epoch.

.. note::

    The `DataLoader` object supports key features such as:
    
    - **Batching**: Returns multiple samples at a time as a batch.
    - **Shuffling**: Randomizes the sample order at the start of each epoch.
    - **Parallel Loading**: Enables multi-process data loading for large datasets.

Custom Data Pipelines
---------------------

The `lucid.data` package allows users to define custom datasets and combine them with 
`DataLoader` for seamless integration.

This enables users to load large datasets from files, preprocess data on-the-fly, 
and create complex data pipelines.

.. hint::

    Use `Dataset` to specify how individual samples are accessed and processed, 
    then wrap it with `DataLoader` to enable batching and parallel loading.

.. admonition:: Example

    Here’s a more advanced usage of `Dataset` and `DataLoader`, where we load images from disk:

    .. code-block:: python

        >>> import lucid
        >>> import lucid.data as data
        >>> import os
        >>> import imageio
        >>>
        >>> class ImageDataset(data.Dataset):
        ...     def __init__(self, image_dir):
        ...         self.image_paths = [
        ...             os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
        ...         ]
        ...     
        ...     def __len__(self):
        ...         return len(self.image_paths)
        ...     
        ...     def __getitem__(self, idx):
        ...         return imageio.imread(self.image_paths[idx])
        
        >>> dataset = ImageDataset('/path/to/images')
        >>> loader = data.DataLoader(dataset, batch_size=4, shuffle=True)
        >>> for batch in loader:
        ...     print(batch.shape)
    
    In this example, `ImageDataset` loads images from a directory, and `DataLoader` handles 
    batching and shuffling the data.

Integration with `lucid`
------------------------

The `data` package integrates seamlessly with other components of the `lucid` library:

- **Tensors**: `DataLoader` returns batches as `Tensor` objects, ready to be fed into models.
- **Neural Networks**: The `DataLoader` makes it easy to iterate through training samples 
  during model training.

.. warning::

    When working with large datasets, be mindful of memory usage.
    Loading entire datasets into memory may cause out-of-memory errors.
    Use on-the-fly data loading via `Dataset` and `DataLoader` to avoid this issue.

Conclusion
----------

The `lucid.data` package provides a comprehensive set of tools for defining and loading datasets.
With `Dataset` and `DataLoader`, users can create efficient, modular, and scalable data pipelines 
for machine learning models.

By utilizing `Dataset` for data access logic and `DataLoader` for batching and shuffling, 
users can streamline the process of feeding data into models.

.. attention::

    For more advanced use cases, consider combining `DataLoader` with preprocessing logic inside `Dataset`.
    This approach allows for dynamic data augmentation and efficient memory usage during training.

