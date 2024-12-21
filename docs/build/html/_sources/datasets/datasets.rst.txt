lucid.datasets
==============

The `lucid.datasets` package provides a collection of predefined datasets 
for use in deep learning models. These datasets span a variety of domains, 
including image, text, and tabular data, offering convenient access to widely-used 
datasets with minimal setup.

Overview
--------

The `lucid.datasets` package simplifies the process of working with datasets by offering:

- **Predefined Datasets**: Popular datasets are available out-of-the-box.
- **Flexible Data Loading**: Support for custom transformations and splitting into training and test sets.
- **Multi-domain Support**: Includes datasets for images, text, and tabular data.
- **Integration with Lucid**: Fully compatible with `lucid.Tensors` for seamless use in neural networks.

Key Features
------------

- **Ease of Use**: Simple and consistent APIs for accessing datasets.
- **Custom Transformations**: Apply preprocessing and augmentations via user-defined `transform` functions.
- **Preset Splits**: Automatically provides training and testing splits where applicable.
- **Extendable**: Users can easily add their own datasets to the framework.

Usage
-----

The `lucid.datasets` package allows for easy access to datasets through predefined classes. 
Each class provides flexible arguments to tailor the dataset to specific needs.

**Example: Loading the MNIST Dataset**

.. code-block:: python

    from lucid.datasets import MNIST

    # Load the training set, downloading it if necessary
    mnist_train = MNIST(root="./data", train=True, download=True)

    # Access a single data point
    image, label = mnist_train[0]

    print(f"Image Shape: {image.shape}, Label: {label}")

Future Work
-----------

The `lucid.datasets` package aims to expand its offerings to include:

- **Text Datasets**: Sentiment analysis, language modeling, and more.
- **Tabular Datasets**: Datasets for regression and classification tasks.

For the latest updates and additional datasets, refer to the Lucid documentation.

.. note::

  - All datasets are designed to work seamlessly with `lucid.Tensors`.
  - Datasets requiring external downloads will store data in the specified `root` directory.
