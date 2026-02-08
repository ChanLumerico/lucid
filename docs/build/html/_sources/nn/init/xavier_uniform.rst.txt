nn.init.xavier_uniform
======================

.. autofunction:: lucid.nn.init.xavier_uniform

The `xavier_uniform` function initializes the input `tensor` with values sampled from a uniform distribution
within a specific range to maintain a stable variance of activations across layers. This initialization method is commonly used
with activation functions like sigmoid and tanh.

Function Signature
------------------

.. code-block:: python

    def xavier_uniform(tensor: Tensor, gain: _Scalar = 1.0) -> None

Parameters
----------

- **tensor** (:class:`Tensor`): 
  The tensor to be initialized. The shape of the tensor determines the fan-in and 
  fan-out for the initialization.

- **gain** (`_Scalar`, optional): 
  An optional scaling factor applied to the computed bound. Defaults to 1.0.

Returns
-------

- **None**: 
  The function modifies the `tensor` in-place with new values sampled from 
  the uniform distribution.

Forward Calculation
-------------------

The values in the tensor are sampled from a uniform distribution 
:math:`U(-\text{bound}, \text{bound})`, where the bound is calculated as:

.. math::

    \text{bound} = \text{gain} \cdot \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

Where:

- :math:`\text{fan\_in}` is the number of input units in the weight tensor.
- :math:`\text{fan\_out}` is the number of output units in the weight tensor.

Backward Gradient Calculation
-----------------------------
Since `xavier_uniform` is an initialization function, it does not have a gradient of its own. 
However, if the tensor is used in subsequent operations that require gradients, 
those gradients will flow backward as part of the computation graph.

Examples
--------

**Basic Xavier Uniform Initialization**

.. code-block:: python

    >>> import lucid
    >>> from lucid import xavier_uniform
    >>> tensor = lucid.zeros((3, 2))
    >>> xavier_uniform(tensor)
    >>> print(tensor)
    Tensor([[ 0.123, -0.234],
            [ 0.342, -0.678],
            [ 0.678,  0.123]], requires_grad=False)

**Xavier Uniform Initialization with Gain**

.. code-block:: python

    >>> tensor = lucid.zeros((4, 4))
    >>> xavier_uniform(tensor, gain=2.0)
    >>> print(tensor)
    Tensor([[ 0.563, -0.342,  0.421, -0.678],
            [-0.321,  0.654, -0.276,  0.345],
            [ 0.876,  0.124, -0.563, -0.234],
            [ 0.543, -0.234,  0.657, -0.421]], requires_grad=False)

.. note::

    - Xavier initialization is best suited for layers with symmetric 
      activation functions such as tanh or sigmoid.
    - For ReLU activations, consider using **Kaiming Initialization** instead 
      for better performance.
