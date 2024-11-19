lucid.nn
========

The `lucid.nn` package provides foundational tools for constructing and managing neural networks. 
It includes essential components such as modules and parameters, enabling users to build flexible 
and efficient deep learning models.

Overview
--------

The `lucid.nn` package is designed to facilitate the development of neural networks by providing:

- **Parameter Management**: Tools for defining and handling trainable parameters.

- **Model Abstraction**: The `Module` class for encapsulating layers and sub-models, 
  promoting modularity and reusability.

- **Integration**: Seamless interaction with other components of the `lucid` library for 
  tensor operations, gradient computation, and more.

Key Components
--------------

.. rubric:: `Parameter`

A `Parameter` is a wrapper for tensors that require gradient computation. 
It is used to define weights, biases, or any other trainable components in a neural network.

.. important::

    Any tensor wrapped in a `Parameter` will be treated as trainable during optimization. 
    Use it wisely for variables that require gradients.

.. admonition:: Example

    The following example demonstrates how to create and use a `Parameter`:

    .. code-block:: python

        >>> import lucid
        >>> import lucid.nn as nn
        >>> weight = nn.Parameter(lucid.zeros((3, 3)))
        >>> print(weight.requires_grad)
        True

    Here, `weight` is a trainable parameter initialized to zeros.

.. caution::

    Ensure all trainable parameters are properly registered under the parent `Module`. 
    Failure to do so might result in missed gradients during backpropagation.

.. rubric:: `Module`

The `Module` class serves as the base class for all neural network models. 
It provides functionality for organizing parameters, defining forward passes, and composing layers.

.. tip::

    Use the `Module` class to create reusable layers and models. 
    This modular design allows for better organization and readability.

.. admonition:: Example

    A custom neural network can be defined by subclassing `Module`:

    .. code-block:: python

        >>> class MyModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.weight = nn.Parameter(lucid.ones((3, 3)))
        ...     def forward(self, x):
        ...         return x @ self.weight

        >>> model = MyModel()
        >>> input_tensor = lucid.Tensor([[1, 2, 3]])
        >>> output = model.forward(input_tensor)
        >>> print(output)

    This example demonstrates a simple model with a learnable weight matrix.

.. note::

    All `Module` objects have built-in support for:
    
    - **Parameter Registration**: Automatically tracks parameters for gradient updates.
    - **Hierarchical Structure**: Allows nesting of submodules for complex architectures.

Organizing Models
-----------------

The `lucid.nn` package supports hierarchical model structures. 
You can define complex models by combining multiple `Module` objects, 
each encapsulating its own parameters and logic.

.. hint::

    Use submodules to break down large models into smaller, manageable parts. 

.. admonition:: Example

    Consider this hierarchical organization of a neural network:

    .. code-block:: python

        >>> class ConvBlock(nn.Module):
        ...     def __init__(self, in_channels, out_channels):
        ...         super().__init__()
        ...         self.weight = nn.Parameter(lucid.randn(out_channels, in_channels, 3, 3))
        ...     def forward(self, x):
        ...         return nn.conv2d(x, self.weight)

        >>> class MyModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.block1 = ConvBlock(3, 16)
        ...         self.block2 = ConvBlock(16, 32)
        ...     def forward(self, x):
        ...         x = self.block1(x)
        ...         return self.block2(x)

        >>> model = MyModel()
        >>> print(model)

Integration with `lucid`
------------------------

The `nn` package integrates seamlessly with other components of the `lucid` library:

- **Tensors**: Use `Tensor` objects for inputs, outputs, and internal computations.
- **Gradient Computation**: Leverages `autograd` features of the library for automatic differentiation.

.. warning::

    Make sure your input tensors are compatible with the shapes expected by 
    your network layers. 
    Shape mismatches will raise runtime errors during forward passes.

Conclusion
----------

The `lucid.nn` package is the core of the library’s neural network functionality, 
offering intuitive abstractions for building and training models. 

By utilizing `Parameter` and `Module`, users can create robust and scalable architectures 
while benefiting from the library’s advanced computational features.

.. attention::

    For more advanced use cases, explore combining `Module` with custom tensor operations 
    from the core `lucid` library. This allows you to create tailored models that fit 
    specific research or production needs.