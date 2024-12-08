lucid.nn.Identity
==================

.. autoclass:: lucid.nn.Identity
    :members:
    :undoc-members:
    :show-inheritance:

The `Identity` module is a neural network layer that performs the identity transformation. 
It returns the input tensor unchanged. This module is useful as a placeholder or when a 
no-operation layer is required within a network architecture.

Class Signature
---------------

.. code-block:: python

    class lucid.nn.Identity(*args, **kwargs)

Parameters
----------

- **args** (*tuple, optional*):
    Variable length argument list. Not used in `Identity` but accepted for compatibility.
    
- **kwargs** (*dict, optional*):
    Arbitrary keyword arguments. Not used in `Identity` but accepted for compatibility.

Attributes
----------

- **None**

Forward Calculation
-------------------

The `Identity` module performs the following operation:

.. math::

    \mathbf{y} = \mathbf{x}

Where:

- :math:`\mathbf{x}` is the input tensor.
- :math:`\mathbf{y}` is the output tensor, identical to the input.

Backward Gradient Calculation
-----------------------------

During backpropagation, the gradient with respect to the input is passed unchanged:

.. math::

    \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \mathbf{I}

Where:

- :math:`\mathbf{I}` is the identity matrix.

This means that the gradient of the loss with respect to the input is the 
same as the gradient of the loss with respect to the output.

Examples
--------

Using the `Identity` module in a neural network:

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)  # Shape: (1, 3)
    >>> identity = nn.Identity()
    >>> output = identity(input_tensor)
    >>> print(output)
    Tensor([[1.0, 2.0, 3.0]], grad=None)
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    Tensor([[1.0, 1.0, 1.0]])

Using `Identity` as a placeholder in a sequential model:

.. code-block:: python

    >>> import lucid.nn as nn
    >>> model = nn.Sequential(
    ...     nn.Linear(3, 5),
    ...     nn.ReLU(),
    ...     nn.Identity(),  # Placeholder for potential future layers
    ...     nn.Linear(5, 2)
    ... )
    >>> input_data = Tensor([[0.5, -1.2, 3.3]], requires_grad=True)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[...]], grad=None)  # Output tensor after passing through the model

