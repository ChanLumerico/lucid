transforms.ToTensor
===================

.. autoclass:: lucid.transforms.ToTensor

The `ToTensor` class converts an input image or array-like structure to a Lucid Tensor. 
This is one of the most commonly used transformations, especially for preparing images 
for deep learning models. It ensures that the input is converted to a format that is 
compatible with Lucid's neural network operations.

Class Signature
---------------

.. code-block:: python

    class ToTensor(nn.Module):
        def __init__(
            self,
            requires_grad: bool = False,
            keep_grad: bool = False,
            dtype: type | None = None
        ) -> None

Parameters
----------

- **requires_grad** (*bool*, optional): 
  If True, gradients will be tracked for the tensor. Default is False.
  This is useful if the resulting tensor will participate in a computation 
  graph where gradients are required.
    
- **keep_grad** (*bool*, optional): 
  If True, gradients will be retained for the tensor after each backward 
  pass. Default is False. This is often used in scenarios where gradients 
  need to be reused.
    
- **dtype** (*type | None*, optional): 
  Specifies the data type of the tensor. If None, the default data type 
  of Lucid tensors (typically `np.float32`) will be used.

Usage
-----

The `ToTensor` class is used to convert image data (such as NumPy arrays or PIL images) 
to Lucid Tensors. It ensures that the data is in the correct format for processing 
by neural networks. This transformation is often the first step in a preprocessing 
pipeline.

Example Usage
-------------

Here is an example of using `ToTensor` to convert an image to a Lucid tensor.

.. code-block:: python

    import lucid.transforms as T
    
    transform = T.ToTensor()
    
    # Convert a NumPy array to a Lucid Tensor
    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    tensor = transform(image)
    
    print(tensor)
    # Output: Tensor([...])

Another example with the `requires_grad` parameter set to True:

.. code-block:: python

    import lucid.transforms as T
    
    transform = T.ToTensor(requires_grad=True)
    
    image = np.array([[1, 2], [3, 4]])
    tensor = transform(image)
    
    print(tensor.requires_grad)  # True

.. note::

    - The `ToTensor` class is essential for converting input data into Lucid Tensors, 
      which are used throughout Lucid's neural network layers.
    - If `requires_grad=True`, the resulting tensor will be part of the computation graph, 
      allowing for automatic gradient computation during backpropagation.
    - By specifying `dtype`, users can control the type of the resulting tensor 
      (for example, `np.float32`, `np.float64`, etc.).

Conclusion
----------

The `ToTensor` class is a fundamental part of the `lucid.transforms` package, 
allowing users to convert various input formats into Lucid tensors. 
With support for gradient tracking, gradient retention, and data type control, 
`ToTensor` serves as a flexible and powerful transformation that is essential 
for deep learning preprocessing pipelines.
