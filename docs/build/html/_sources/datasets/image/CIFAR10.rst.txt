CIFAR10
=======

.. autoclass:: lucid.datasets.CIFAR10

The `CIFAR10` class provides access to the CIFAR-10 dataset, 
a widely-used benchmark in the field of image classification. 

This dataset contains 60,000 RGB images (32x32 pixels) across 10 classes, 
with 50,000 training images and 10,000 test images.

Class Signature
---------------

.. code-block:: python

    class lucid.datasets.CIFAR10(
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

    from lucid.datasets import CIFAR10

    # Load the training set, downloading it if necessary
    cifar10_train = CIFAR10(root="./data", train=True, download=True)

    # Get the first image and label
    image, label = cifar10_train[0]

    print(f"Image Shape: {image.shape}, Label: {label}")

**Applying Transformations**

.. code-block:: python

    from lucid.datasets import CIFAR10
    from lucid.transforms import Normalize

    transform = Normalize(mean=[...], std=[...])

    cifar10_train = CIFAR10(root="./data", train=True, download=True, transform=transform)
    image, label = cifar10_train[0]

    print(f"Normalized Image: {image}")

.. note::

    - The dataset is stored in `.npy` format for efficient loading.
    - Integration with OpenML ensures reliable access to the dataset.

References
----------

- Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. 
  "The CIFAR-10 dataset." [🔗](https://www.cs.toronto.edu/~kriz/cifar.html)
