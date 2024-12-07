lucid.nn.Module  
===============

.. autoclass:: lucid.nn.Module

The `Module` class is a foundational building block of the `lucid` library. 
It provides a modular interface for creating and managing neural network components, 
handling the registration of submodules and parameters, and maintaining their 
state during training and evaluation.

The `Module` class serves as a base for defining custom neural network layers, 
enabling functionality like forward propagation, parameter management, 
and serialization of model states.

Class Signature  
---------------

.. code-block:: python

    class Module:
        def __init__(self) -> None

Methods  
-------

### Core Methods

- **__setattr__(name: str, value: Any) -> None**:
  Dynamically manages the addition of `Parameter` or `Module` objects as 
  attributes to the module. Ensures that attributes are appropriately 
  stored in `_parameters` or `_modules`.

  **Parameters** 
  - **name** (*str*): The name of the attribute to be added.  
  - **value** (*Any*): The attribute value, which could be a `Parameter`, `Module`, or other type.

- **add_module(name: str, module: Self) -> None**:
  Adds a submodule to the current module. Ensures that the submodule is registered correctly.

  **Parameters**
  - **name** (*str*): The name of the submodule.  
  - **module** (*Module*): The submodule instance to add.

  **Raises**
  - **TypeError**: If the `module` is not an instance of `Module` or `None`.

- **register_parameter(name: str, param: Parameter) -> None**:
  Registers a parameter to the module. The parameter becomes part of the model's 
  learnable parameters.

  **Parameters**  
  - **name** (*str*): The name of the parameter.  
  - **param** (*Parameter*): The `Parameter` instance to register.

  **Raises**  
  - **TypeError**: If `param` is not a `Parameter` or `None`.

- **forward() -> Tensor | tuple[Tensor, ...]**: 
  Placeholder for the forward pass. Must be implemented by subclasses.

  **Returns**  
  - A `Tensor` or a tuple of `Tensor` objects representing the output of the forward pass.

  **Raises**  
  - **NotImplementedError**: If not overridden by the subclass.

### Utilities

- **parameters(recurse: bool = True) -> Iterator**:
  Returns an iterator over all parameters in the module. 
  Includes parameters from submodules if `recurse` is `True`.

  **Parameters**  
  - **recurse** (*bool*, optional): Whether to include parameters from submodules. Defaults to `True`.

  **Yields**  
  - **Parameter**: The parameters in the module.

- **modules() -> Iterator**:
  Returns an iterator over all submodules, including the current module.

  **Yields**
  - **Module**: The submodules of the current module.

- **state_dict(destination: OrderedDict | None = None, prefix: str = \"\", keep_vars: bool = False) -> dict[str, Parameter]**: 
  Returns a dictionary containing the state of the module, including parameters and submodules.

  **Parameters**:  
  - **destination** (*OrderedDict | None*, optional): The destination dictionary to populate. Defaults to a new `OrderedDict`.  
  - **prefix** (*str*, optional): A prefix to prepend to parameter names. Defaults to an empty string.  
  - **keep_vars** (*bool*, optional): Whether to keep the variable references. Defaults to `False`.

  **Returns**:  
  - **dict[str, Parameter]**: A dictionary mapping parameter names to their values.

- **load_state_dict(state_dict: dict[str, Parameter], strict: bool = True) -> None**:  
  Loads parameters from a `state_dict`. Matches parameters by name and assigns them to the module.

  **Parameters**:  
  - **state_dict** (*dict[str, Parameter]*): A dictionary containing the parameters to load.  
  - **strict** (*bool*, optional): Whether to enforce an exact match between `state_dict` and the module. Defaults to `True`.

  **Raises**:  
  - **KeyError**: If there are missing or unexpected keys in `state_dict` when `strict` is `True`.

### Special Methods

- **__call__(*args: Any, **kwargs: Any) -> Tensor | tuple[Tensor, ...]**:  
  Calls the `forward` method of the module, passing the provided arguments and keyword arguments.

  **Parameters**:  
  - **args**: Positional arguments to pass to the `forward` method.  
  - **kwargs**: Keyword arguments to pass to the `forward` method.

  **Returns**:  
  - **Tensor | tuple[Tensor, ...]**: The result of the `forward` method.

Examples  
--------

.. admonition:: **Defining a custom module**

    .. code-block:: python

        import lucid.nn as nn

        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.param1 = nn.Parameter([1.0, 2.0, 3.0])
                self.submodule = nn.Module()

            def forward(self, x):
                # Perform operations here
                return x * self.param1

        model = MyModel()
        print(model)

.. tip:: **Inspecting parameters and submodules**  

    Use the `parameters()` and `modules()` methods to inspect the components of a model.

    .. code-block:: python

        for param in model.parameters():
            print(param)

        for submodule in model.modules():
            print(submodule)

.. warning:: **State dictionary mismatch**  

    When loading a state dictionary, ensure the keys match the module's structure. If `strict=True`, mismatched keys will raise an error.

    .. code-block:: python

        state_dict = {'param1': nn.Parameter([0.5, 0.5, 0.5])}
        model.load_state_dict(state_dict, strict=False)
