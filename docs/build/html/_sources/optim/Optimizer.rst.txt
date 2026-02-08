optim.Optimizer
===============

.. autoclass:: lucid.optim.Optimizer

The `Optimizer` class is an abstract base class for optimization algorithms in the `lucid` library. 
It provides a framework for updating the parameters of neural network models based on computed gradients. 
Subclasses must implement the `step` method to define the specific optimization logic.

Class Signature
---------------

.. code-block:: python

    class Optimizer(ABC):
        def __init__(
            self, params: Iterable[nn.Parameter], defaults: dict[str, Any]
        ) -> None

Methods
-------

**Core Methods**

.. code-block:: python

    def __init__(
        self, params: Iterable[nn.Parameter], defaults: dict[str, Any]
    ) -> None

Initializes the optimizer with the given parameters and default settings.

**Parameters**:

- **params** (*Iterable[nn.Parameter]*): 
  An iterable of `Parameter` instances to be optimized.

- **defaults** (*dict[str, Any]*): 
  A dictionary of default hyperparameters for the optimizer (e.g., learning rate).

**Raises**:

- **TypeError**: If `params` is not an iterable of `Parameter` instances.

.. code-block:: python

    def param_groups_setup(
        self, params: list[nn.Parameter], defaults: dict[str, Any]
    ) -> list[dict[str, Any]]
    
Sets up parameter groups with the provided parameters and default settings.

**Parameters**:

- **params** (*list[nn.Parameter]*): 
  A list of `Parameter` instances to include in the optimizer.

- **defaults** (*dict[str, Any]*): 
  A dictionary of default hyperparameters for the optimizer.

**Returns**:

- **list[dict[str, Any]]**: 
  A list containing a single parameter group dictionary.

.. code-block:: python

    @abstractmethod
    def step(self, closure: Callable[[], Any] | None = None) -> Any | None:
        raise NotImplementedError(...)

Performs a single optimization step, updating the parameters based on computed gradients.

**Parameters**:

- **closure** (*Callable[[], Any] | None*, optional): 
  A closure that reevaluates the model and returns the loss. Defaults to `None`.

**Returns**:

- **Any | None**: 
  The result of the closure if provided, otherwise `None`.

**Raises**:

- **NotImplementedError**: If not overridden by the subclass.

.. code-block:: python

    def zero_grad(self) -> None

Sets the gradients of all optimized parameters to zero.

.. code-block:: python

    def add_param_group(self, param_group: dict[str, Any]) -> None

Adds a new parameter group to the optimizer.

**Parameters**:

- **param_group** (*dict[str, Any]*): 
  A dictionary specifying a parameter group, containing a `"params"` key with a 
  list of `Parameter` instances and other optimizer-specific settings.

**Raises**:

- **ValueError**: If any parameter appears in more than one parameter group.

.. code-block:: python

    def state_dict(self) -> _StateDict

Returns the state of the optimizer, including parameter states and parameter groups.

**Parameters**:
    
- **None**

**Returns**:
    
- **_StateDict**: A dictionary containing the optimizer's state and parameter groups.

.. code-block:: python

    def load_state_dict(self, state_dict: _StateDict) -> None

Loads the optimizer state from a `state_dict`.

**Parameters**:

- **state_dict** (*_StateDict*): 
  A dictionary containing the optimizer state and parameter groups to load.

**Returns**:
    
- **None**

.. code-block:: python

    def __repr__(self) -> str

Returns a string representation of the optimizer, including its parameter groups.

**Parameters**:
    
- **None**

**Returns**:
    
- **str**: A string representing the optimizer.

Examples
--------

.. admonition:: **Defining a custom optimizer**
   :class: note

   .. code-block:: python

       import lucid.optim as optim
       import lucid.nn as nn

       class MyOptimizer(optim.Optimizer):
           def __init__(self, params, lr=0.01):
               defaults = {'lr': lr}
               super().__init__(params, defaults)

           def step(self, closure=None):
               for group in self.param_groups:
                   for param in group['params']:
                       if param.grad is not None:
                           param.data -= group['lr'] * param.grad

       # Usage
       model = nn.Module()
       # Assume model has parameters
       optimizer = MyOptimizer(model.parameters(), lr=0.01)

.. tip:: **Inspecting optimizer state**

    Use the `state_dict()` and `load_state_dict()` methods to save and 
    load the optimizer state.

    .. code-block:: python

        # Save state
        optimizer_state = optimizer.state_dict()

        # Load state
        optimizer.load_state_dict(optimizer_state)

.. warning:: **State dictionary mismatch**

    When loading a state dictionary, ensure the keys match the optimizer's structure. 
    If mismatched and `strict=True`, an error will be raised.
