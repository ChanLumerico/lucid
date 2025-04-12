lucid.arange
============

.. autofunction:: lucid.arange

The `arange` function creates a tensor with evenly spaced values within a specified interval.

Function Signature
------------------

.. code-block:: python

    def arange(
        start: _Scalar = 0.0,
        stop: _Scalar,
        step: _Scalar = 1.0,
        dtype: Any = np.float32,
        requires_grad: bool = False,
        keep_grad: bool = False,
        device: _DeviceType = "cpu",
    ) -> Tensor

Parameters
----------

- **start** (*_Scalar*, optional): 
  The starting value of the sequence. Defaults to `0.0` if only `stop` is provided.

- **stop** (*_Scalar*): 
  The end value of the sequence (exclusive).

- **step** (*_Scalar*, optional): 
  The spacing between values. Defaults to `1.0`.

- **dtype** (*Any*, optional): 
  The data type of the elements in the tensor. Defaults to `np.float32`.

- **requires_grad** (*bool*, optional): 
  If `True`, the resulting tensor will be part of the computation graph and 
  capable of tracking gradients. Defaults to `False`.

- **keep_grad** (*bool*, optional): 
  If `True`, the gradient history will be preserved even if the tensor does not 
  require gradients. Defaults to `False`.

Returns
-------

- **Tensor**: A tensor containing evenly spaced values within the specified interval.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = lucid.arange(5)
    >>> print(a)
    Tensor([0. 1. 2. 3. 4.])

    >>> a = lucid.arange(2, 5)
    >>> print(a)
    Tensor([2. 3. 4.])

    >>> a = lucid.arange(0, 1, 0.2)
    >>> print(a)
    Tensor([0.  0.2 0.4 0.6 0.8])

The `arange` function also allows specifying `requires_grad`:

.. code-block:: python

    >>> a = lucid.arange(0, 5, 1, requires_grad=True)
    >>> print(a.requires_grad)
    True

.. note::

    - This function is often used to generate sequences of numbers for indexing, 
      iteration, or initialization purposes in neural networks and other numerical applications.

    - The `requires_grad` parameter is useful for differentiable operations 
      in computation graphs.

    - If `keep_grad` is set to `True`, the tensor will retain its gradient 
      history even if `requires_grad` is `False`.
