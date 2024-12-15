nn.init.xavier_normal_
======================

.. autofunction:: lucid.nn.init.xavier_normal_

The `xavier_normal_` function initializes the input `tensor` with values sampled from a normal distribution 
:math:`\mathcal{N}(0, \sigma^2)`, where the standard deviation :math:`\sigma` is calculated 
to maintain a stable variance of activations across layers. 

This initialization method is commonly used with activation functions like sigmoid and tanh.

Function Signature
------------------

.. code-block:: python

    def xavier_normal_(tensor: Tensor, gain: _Scalar = 1.0) -> None

Parameters
----------

- **tensor** (:class:`Tensor`): 
  The tensor to be initialized. The shape of the tensor determines the fan-in and 
  fan-out for the initialization.

- **gain** (`_Scalar`, optional): 
  An optional scaling factor applied to the computed standard deviation. Defaults to 1.0.

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

    \sigma = \text{gain} \cdot \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

Where:

- :math:`\text{fan\_in}` is the number of input units in the weight tensor.
- :math:`\text{fan\_out}` is the number of output units in the weight tensor.

Examples
--------

**Basic Xavier Normal Initialization**

.. code-block:: python

    >>> import lucid
    >>> from lucid.nn.init import xavier_normal_
    >>> tensor = lucid.zeros((3, 2))
    >>> xavier_normal_(tensor)
    >>> print(tensor)
    Tensor([[ 0.123, -0.234],
            [ 0.342, -0.678],
            [ 0.678,  0.123]], requires_grad=False)

**Xavier Normal Initialization with Gain**

.. code-block:: python

    >>> tensor = lucid.zeros((4, 4))
    >>> xavier_normal_(tensor, gain=2.0)
    >>> print(tensor)
    Tensor([[ 0.563, -0.342,  0.421, -0.678],
            [-0.321,  0.654, -0.276,  0.345],
            [ 0.876,  0.124, -0.563, -0.234],
            [ 0.543, -0.234,  0.657, -0.421]], requires_grad=False)

.. note::
    
    - Xavier initialization is best suited for layers with symmetric activation functions 
      such as tanh or sigmoid.
    - For ReLU activations, consider using **Kaiming Initialization** instead for better 
      performance.

.. seealso::

    - :func:`xavier_uniform_`
    - :func:`kaiming_uniform_`
    - :func:`kaiming_normal_`
