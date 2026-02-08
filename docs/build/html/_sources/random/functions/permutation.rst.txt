lucid.random.permutation
========================

.. autofunction:: lucid.random.permutation

The `permutation` function returns a **random permutation** of integers from 
:math:`0` to :math:`n - 1`, without repetition. It produces a 1-dimensional tensor 
containing all integers in a shuffled order.

Function Signature
------------------

.. code-block:: python

    def permutation(
        n: int,
        dtype: type = np.int64,
        requires_grad: bool = False,
        keep_grad: bool = False,
        device: _DeviceType = "cpu",
    ) -> Tensor

Parameters
----------

- **n** (*int*):  
  The number of elements to permute. The output tensor will contain 
  all integers from `0` to `n - 1`, randomly shuffled.

- **dtype** (*type*, optional):  
  The data type of the resulting tensor. Defaults to `lucid.Int64`.

- **requires_grad** (*bool*, optional):  
  Whether the resulting tensor should track gradients. Defaults to `False`.

- **keep_grad** (*bool*, optional):  
  Whether to preserve gradient history during further operations. Defaults to `False`.

- **device** (*{ "cpu", "gpu" }*, optional):  
  The device on which to place the resulting tensor. Defaults to `"cpu"`.

Returns
-------

- **Tensor**:  
  A 1-dimensional tensor of length `n`, containing a random permutation of 
  integers from `0` to `n - 1`.

.. math::

   \operatorname{shape}(\text{out}) = (n,)

Example
-------

.. code-block:: python

    >>> import lucid
    >>> lucid.random.permutation(5)
    Tensor([3, 0, 2, 4, 1], grad=None)

The permutation contains all integers from 0 to 4 in a random order.

To place the tensor on GPU:

.. code-block:: python

    >>> lucid.random.permutation(6, device="gpu")
    Tensor([4, 0, 2, 1, 5, 3], device=gpu)

.. note::

    - The permutation is generated using NumPy and transferred to the specified device.
    - Values are unique and cover the full range from 0 to :math:`n-1`.
    - This operation is **non-differentiable**, and gradients will not flow 
      through the result.

    - For reproducibility, use :func:`lucid.random.seed` before calling this function.
