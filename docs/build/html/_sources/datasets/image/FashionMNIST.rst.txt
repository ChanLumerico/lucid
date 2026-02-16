FashionMNIST
============

.. autoclass:: lucid.datasets.FashionMNIST

The `FashionMNIST` class provides access to the Fashion-MNIST dataset, 
which serves as a more challenging replacement for the traditional MNIST dataset. 
It contains images of various clothing items rather than handwritten digits.

This dataset consists of 70,000 grayscale images (28x28 pixels) 
representing 10 categories of clothing items, including labels for classification.

Class Signature
---------------

.. code-block:: python

    class lucid.datasets.FashionMNIST(
        root: str | Path,
        train: bool = True,
        download: bool = False,
        transform: nn.Module | Compose | None = None,
        target_transform: nn.Module | Compose | None = None,
        test_size: float = 0.2,
        to_tensor: bool = True,
    )

Parameters
----------

- **root** (*str | Path*):
  The directory where the dataset will be stored.

- **train** (*bool*, optional):
  If True, loads the training set (80% of the dataset by default). 
  If False, loads the test set (remaining 20%). Defaults to True.

- **download** (*bool*, optional):
  If True, downloads the dataset from OpenML if it is not available 
  in the specified `root` directory. Defaults to False.

- **transform** (*nn.Module | Compose | None*, optional):
  A function or transform pipeline to apply to the images. Defaults to None.

- **target_transform** (*nn.Module | Compose | None*, optional):
  A function or transform pipeline to apply to the labels. Defaults to None.

- **test_size** (*float*, optional):
  The proportion of the dataset to be used as the test set. Defaults to 0.2.

- **to_tensor** (*bool*, optional):
  If True, converts the data into `lucid.Tensors`. Defaults to True.

Attributes
----------

- **data** (*lucid.Tensor*): A tensor containing the images in the dataset.
- **targets** (*lucid.Tensor*): A tensor containing the labels corresponding to the images.

Methods
-------

- **__getitem__(index: int) -> Tuple[Tensor, Tensor]:**
  Returns a tuple containing the image and label at the specified index.

- **__len__() -> int:**
  Returns the total number of samples in the dataset.

Examples
--------

**Loading and Accessing Data**

.. code-block:: python

    from lucid.datasets import FashionMNIST

    # Load the training set, downloading it if necessary
    fashion_train = FashionMNIST(root="./data", train=True, download=True)

    # Get the first image and label
    image, label = fashion_train[0]

    print(f"Image Shape: {image.shape}, Label: {label}")

**Applying Transformations**

.. code-block:: python

    from lucid.datasets import FashionMNIST
    from lucid.transforms import Normalize

    transform = Normalize(mean=[...], std=[...])

    fashion_train = FashionMNIST(root="./data", train=True, download=True, transform=transform)
    image, label = fashion_train[0]

    print(f"Normalized Image: {image}")

.. note::

    - The dataset is stored in `.npy` format for efficient loading.
    - Integration with OpenML ensures reliable access to the dataset.

References
----------

- Han Xiao, Kashif Rasul, and Roland Vollgraf. 
  "Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms." 
  [🔗](https://github.com/zalandoresearch/fashion-mnist)
