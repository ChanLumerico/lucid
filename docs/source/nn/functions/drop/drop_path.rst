nn.functional.drop_path
=======================

.. autofunction:: lucid.nn.functional.drop_path

The `drop_path` function implements stochastic depth, a regularization technique 
commonly used in neural networks to improve generalization. It randomly drops entire 
paths (or layers) in the network during training by zeroing out elements of the 
input tensor. The remaining elements can optionally be scaled by the keep probability 
to preserve the overall magnitude of the input.

Function Signature
------------------

.. code-block:: python

    def drop_path(input_: Tensor, p: float = 0.1, scale_by_keep: bool = True) -> Tensor

Parameters
----------

- **input_** (*Tensor*):
  The input tensor that will undergo stochastic depth.

- **p** (*float*, optional):
  Probability of dropping a path (element). Default is 0.1. Must be in the range [0, 1].

- **scale_by_keep** (*bool*, optional):
  If `True`, the input is scaled by `1 / (1 - p)` to maintain the expected value. 
  Default is `True`.

Returns
-------

- **Tensor**:
  A tensor of the same shape as `input_`, where some elements are zeroed out 
  according to the probability `p`. If `scale_by_keep` is enabled, the remaining 
  elements are scaled appropriately.

Forward Calculation
-------------------

The forward operation for `drop_path` is defined as:

.. math::

    \text{output}_i = \begin{cases}
        0, & \text{with probability } p, \\
        \frac{\text{input}_i}{1 - p}, & \text{otherwise, if scale_by_keep is True}, \\
        \text{input}_i, & \text{otherwise, if scale_by_keep is False}.
    \end{cases}

This operation is applied element-wise.

Examples
--------

**Basic Usage**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn.functional as F
    >>> input_tensor = lucid.Tensor([1.0, 2.0, 3.0, 4.0])
    >>> output = F.drop_path(input_tensor, p=0.25)
    >>> print(output)
    Tensor([...], grad=None)  # Some elements will be zeroed out

**With `scale_by_keep` Disabled**

.. code-block:: python

    >>> output = F.drop_path(input_tensor, p=0.5, scale_by_keep=False)
    >>> print(output)
    Tensor([...], grad=None)  # No scaling is applied to remaining elements

**Using in a Neural Network**

.. code-block:: python

    >>> class SimpleBlock(lucid.nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.dense = lucid.nn.Linear(4, 4)
    ...     def forward(self, x):
    ...         x = self.dense(x)
    ...         x = F.drop_path(x, p=0.3)
    ...         return x

    >>> block = SimpleBlock()
    >>> input_tensor = lucid.Tensor([[1.0, 2.0, 3.0, 4.0]])
    >>> output = block(input_tensor)
    >>> print(output)
    Tensor([...], grad=None)
