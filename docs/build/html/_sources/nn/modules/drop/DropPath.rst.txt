nn.DropPath
===========

.. autoclass:: lucid.nn.DropPath

The `DropPath` module implements stochastic depth as a reusable layer in neural networks. 
It applies random dropping to entire paths (or layers) in the network during training while 
optionally scaling the remaining elements to preserve the expected value.

Class Signature
---------------

.. code-block:: python

    class DropPath(nn.Module):
        def __init__(self, drop_prob: float = 0.1, scale_by_keep: bool = True) -> None

Parameters
----------

- **drop_prob** (*float*, optional):
  Probability of dropping a path. Default is 0.1. Must be in the range [0, 1].

- **scale_by_keep** (*bool*, optional):
  If `True`, scales the input by `1 / (1 - drop_prob)` to maintain the expected 
  value of the input tensor. Default is `True`.

Attributes
----------

- **drop_prob** (*float*):
  The probability of dropping a path during training.

- **scale_by_keep** (*bool*):
  Whether to scale the input tensor when paths are dropped.

Forward Calculation
-------------------

During the forward pass, the `DropPath` module operates as follows:

.. math::

    \text{output}_i = \begin{cases}
        0, & \text{with probability } \text{drop_prob}, \\
        \frac{\text{input}_i}{1 - \text{drop_prob}}, & \text{otherwise, if scale_by_keep is True}, \\
        \text{input}_i, & \text{otherwise, if scale_by_keep is False}.
    \end{cases}

This operation is applied element-wise to the input tensor.

Examples
--------

**Using DropPath in a Neural Network**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn

    >>> class SimpleModel(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.linear = nn.Linear(4, 4)
    ...         self.drop_path = nn.DropPath(drop_prob=0.2)
    ...     def forward(self, x):
    ...         x = self.linear(x)
    ...         x = self.drop_path(x)
    ...         return x

    >>> model = SimpleModel()
    >>> input_tensor = lucid.Tensor([[1.0, 2.0, 3.0, 4.0]])
    >>> output = model(input_tensor)
    >>> print(output)
    Tensor([...], grad=None)  # Some paths may be zeroed out

**Using DropPath with `scale_by_keep` Disabled**

.. code-block:: python

    >>> drop_path = nn.DropPath(drop_prob=0.5, scale_by_keep=False)
    >>> input_tensor = lucid.Tensor([1.0, 2.0, 3.0, 4.0])
    >>> output = drop_path(input_tensor)
    >>> print(output)
    Tensor([...], grad=None)  # Remaining elements are not scaled

.. note::

  - The `DropPath` module is typically used during training and behaves as an 
    identity mapping during evaluation (`eval` mode).
  - Ensure that `drop_prob` is in the range [0, 1], as values outside this range 
    will result in a runtime error.
