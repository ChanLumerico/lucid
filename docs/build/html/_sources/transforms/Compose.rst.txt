transforms.Compose
==================

.. autoclass:: lucid.transforms.Compose

The `Compose` class allows users to chain multiple transformations together to create a transformation pipeline. 
This enables users to apply several transformations sequentially to an input tensor, making it a powerful tool 
for image preprocessing and data augmentation.

Class Signature
---------------

.. code-block:: python

    class Compose:
        def __init__(self, transforms: List[Callable[[Any], Any]]) -> None

Parameters
----------

- **transforms** (*List[Callable[[Any], Any]]*): 
  A list of transformation functions that are applied sequentially to the input. 
  Each function in the list should take a tensor as input and return a transformed tensor.

Usage
-----

The `Compose` class is used to create a sequence of transformations. It takes a list of transformations 
and applies them in the specified order to the input tensor. This is particularly useful when performing 
data augmentation and preprocessing for deep learning models.

Example Usage
-------------

Here is an example of using `Compose` to create a simple transformation pipeline that converts an image 
to a tensor, resizes it to 128x128, and normalizes it.

.. code-block:: python

    import lucid.transforms as T
    
    transform_pipeline = T.Compose([
        T.ToTensor(),
        T.Resize((128, 128)),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    transformed_image = transform_pipeline(image)

In this example, the `Compose` class chains the transformations together. First, `ToTensor` converts 
the image to a Lucid tensor. Next, `Resize` resizes the image to 128x128, and finally, `Normalize` 
normalizes the image using the given mean and standard deviation.

.. note::

    - The `Compose` class is essential when you want to define reusable transformation pipelines.
    - It is useful for preprocessing datasets, especially for training neural networks where multiple 
      preprocessing steps are needed for each input image.
    - Transformations are applied sequentially, so the order of transformations 
      can significantly affect the output.
