nn.init.kaiming_normal
======================

.. autofunction:: lucid.nn.init.kaiming_normal

The `kaiming_normal` function initializes the input `tensor` with values sampled from a normal distribution 
:math:`\mathcal{N}(0, \sigma^2)`, where the standard deviation :math:`\sigma` is calculated to maintain 
a stable variance of activations in the layer. 

This initialization method is well-suited for layers using ReLU or other non-linear activation functions.

Function Signature
------------------

.. code-block:: python

    def kaiming_normal(tensor: Tensor, mode: _FanMode = "fan_in") -> None

Parameters
----------

- **tensor** (:class:`Tensor`): 
  The tensor to be initialized. The shape of the tensor determines the fan-in and 
  fan-out for the initialization.

- **mode** (`_FanMode`, optional): 
  Determines whether to use `fan_in` or `fan_out` for computing the standard deviation. 
  Defaults to `fan_in`.

Returns
-------

- **None**: 
  The function modifies the `tensor` in-place with new values sampled from 
  the normal distribution.

Forward Calculation
-------------------

The values in the tensor are sampled from a normal distribution 
:math:`\mathcal{N}(0, \sigma^2)`, where the standard deviation :math:`\sigma` 
is calculated as:

.. math::

    \sigma = \sqrt{\frac{2}{\text{fan}}}

Where :math:`\text{fan}` is determined by the `mode` parameter:

- If `mode="fan_in"`, then :math:`\text{fan} = \text{fan\_in}` where :math:`\text{fan\_in}` 
  is the number of input units in the weight tensor.
- If `mode="fan_out"`, then :math:`\text{fan} = \text{fan\_out}` where :math:`\text{fan\_out}` 
  is the number of output units in the weight tensor.

Examples
--------

**Basic Kaiming Normal Initialization**

.. code-block:: python

    >>> import lucid
    >>> from lucid.nn.init import kaiming_normal
    >>> tensor = lucid.zeros((3, 2))
    >>> kaiming_normal(tensor)
    >>> print(tensor)
    Tensor([[ 0.423, -0.234],
            [ 0.342, -0.678],
            [ 0.678,  0.123]], requires_grad=False)

**Kaiming Normal Initialization with fan_out mode**

.. code-block:: python

    >>> tensor = lucid.zeros((4, 4))
    >>> kaiming_normal(tensor, mode="fan_out")
    >>> print(tensor)
    Tensor([[ 0.563, -0.342,  0.421, -0.678],
            [-0.321,  0.654, -0.276,  0.345],
            [ 0.876,  0.124, -0.563, -0.234],
            [ 0.543, -0.234,  0.657, -0.421]], requires_grad=False)

.. note::

    - Kaiming initialization is best suited for layers with ReLU or similar non-linear activations.
    - For layers with tanh or sigmoid activations, consider using **Xavier Initialization** instead for better performance.
