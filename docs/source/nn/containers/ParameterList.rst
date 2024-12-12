nn.ParameterList
================

.. autoclass:: lucid.nn.ParameterList

The `ParameterList` module is a container designed to hold a list of `Parameter` objects. 
It provides an ordered, iterable collection of parameters that can be easily managed and 
integrated into custom neural network modules. This container is particularly useful when 
you need to register multiple parameters that are not associated with any specific sub-module, 
allowing them to be included in the model's parameter list for optimization.

.. warning::

    All elements added to a `ParameterList` must be instances of `Parameter`. 
    Adding non-`Parameter` objects will result in errors.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.ParameterList(parameters: list[nn.Parameter] | None = None) -> None

Parameters
----------
- **parameters** (*list[nn.Parameter]* | *None*, optional):
  An optional list of `Parameter` objects to be added to the `ParameterList` container upon 
  initialization. If provided, each `Parameter` in the list is appended in the order they are passed. 
  Default is `None`.

Attributes
----------
- **parameters** (*List[nn.Parameter]*):
  The internal list that stores all `Parameter` objects added to the `ParameterList` container. 
  Parameters can be accessed by their indices or iterated over during the forward pass.

Forward Calculation
-------------------
The `ParameterList` container itself does not define a specific forward pass. 
Instead, it serves as a collection of parameters that can be accessed and utilized 
within the forward method of a custom `Module`. The forward computation is defined by the user, 
who can retrieve and apply each parameter as needed.

.. math::

    \mathbf{y} = f(\mathbf{x}, \mathbf{p}_1, \mathbf{p}_2, \dots, \mathbf{p}_n)

Where:

- :math:`\mathbf{x}` is the input tensor.
- :math:`\mathbf{p}_1, \mathbf{p}_2, \dots, \mathbf{p}_n` are the parameters contained within the `ParameterList` container.
- :math:`f` is a user-defined function that utilizes the parameters to compute the output.
- :math:`\mathbf{y}` is the final output tensor after applying the parameters.

.. note::

    Unlike `ModuleList`, `ParameterList` is specifically intended for storing parameters 
    rather than modules. It does not handle the forward pass of sub-modules.

Backward Gradient Calculation
-----------------------------
Gradients are automatically handled during backpropagation for each `Parameter` 
contained within the `ParameterList`. Since `ParameterList` is a container, 
it does not directly modify gradients; instead, each `Parameter` computes its own 
gradients based on the loss.

.. note::

    All `Parameter` objects added to a `ParameterList` must support gradient computation for
     backpropagation to function correctly. Failure to do so will prevent the model from learning effectively.

Examples
--------
**Using `ParameterList` within a custom neural network module:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> 
    >>> # Define a custom neural network model using ParameterList
    >>> class CustomParameterModel(nn.Module):
    ...     def __init__(self, input_size, num_parameters):
    ...         super(CustomParameterModel, self).__init__()
    ...         # Initialize a list of parameters
    ...         params = [nn.Parameter(Tensor([1.0], requires_grad=True)) for _ in range(num_parameters)]
    ...         self.param_list = nn.ParameterList(params)
    ...     
    ...     def forward(self, x):
    ...         # Apply each parameter to the input tensor
    ...         for param in self.param_list:
    ...             x = x * param
    ...         return x
    ...
    >>> 
    >>> # Initialize the model with 3 parameters
    >>> model = CustomParameterModel(input_size=1, num_parameters=3)
    >>> 
    >>> # Define input tensor
    >>> input_tensor = Tensor([[2.0]], requires_grad=True)  # Shape: (1, 1)
    >>> 
    >>> # Forward pass
    >>> output = model(input_tensor)
    >>> print(output)
    Tensor([[2.0]], grad=None)  # Output after multiplying by each parameter (1.0 * 1.0 * 1.0)
    >>> 
    >>> # Modify one parameter and perform another forward pass
    >>> model.param_list[0] = nn.Parameter(Tensor([3.0], requires_grad=True))
    >>> output = model(input_tensor)
    >>> print(output)
    Tensor([[6.0]], grad=None)  # Output after multiplying by the updated parameter (3.0 * 1.0 * 1.0)
    >>> 
    >>> # Backpropagation
    >>> loss = output.sum()
    >>> loss.backward()
    >>> print(input_tensor.grad)
    [[1.0]]  # Gradient with respect to input_tensor

.. note::

    Using `ParameterList` allows for the dynamic management of parameters within a model, 
    enabling flexible architectures where the number of parameters can vary. 
    
    This is especially useful in scenarios such as attention mechanisms, 
    where the number of parameters may depend on the input data or model configuration.

.. warning::

    Ensure that all parameters within a `ParameterList` are properly initialized and have 
    `requires_grad` set to `True` if they need to be optimized during training. 
    
    Neglecting to do so will result in those parameters not being updated during 
    the optimization process.

