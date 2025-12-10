lucid.linalg.norm
=================

.. autofunction:: lucid.linalg.norm

The `norm` function computes the :math:`p`-norm of a tensor, 
where :math:`p` is specified by the `ord` parameter.

Function Signature
------------------

.. code-block:: python

    def norm(a: Tensor, ord: int = 2) -> Tensor

Parameters
----------

- **a** (*Tensor*): 
    The input tensor for which the norm is computed.

- **ord** (*int, optional*): 
    The order of the norm. Defaults to :math:`2` (Euclidean norm). 
    
    Supported values include:
    - :math:`1`: Manhattan norm (sum of absolute values).
    - :math:`2`: Euclidean norm.

Returns
-------

- **Tensor**: 
    A scalar tensor representing the computed :math:`p`-norm of the input tensor.

Forward Calculation
-------------------

The :math:`p`-norm is calculated as:

.. math::

    \| \mathbf{a} \|_p = \left( \sum_i |\mathbf{a}_i|^p \right)^{1/p}

For :math:`p = 1`, this becomes the sum of absolute values.  
For :math:`p = 2`, it is the Euclidean norm.  
For :math:`p = \infty`, it is the maximum absolute value.

Backward Gradient Calculation
-----------------------------

The gradient of the :math:`p`-norm with respect to the tensor :math:`\mathbf{a}` is computed as:

.. math::

    \frac{\partial \| \mathbf{a} \|_p}{\partial \mathbf{a}_i} = 
    \begin{cases} 
    \text{sgn}(\mathbf{a}_i) \cdot |\mathbf{a}_i|^{p-1} \cdot \| \mathbf{a} \|_p^{1-p}, & \text{if } p > 1 \\ 
    \text{sgn}(\mathbf{a}_i), & \text{if } p = 1 \\ 
    0, & \text{if } \mathbf{a}_i \neq \max(|\mathbf{a}|) \text{ for } p = \infty 
    \end{cases}

This calculation depends on the value of `ord`.

Raises
------

.. attention::

    - **ValueError**: If the value of `ord` is not supported.
    - **LinAlgError**: If the input tensor does not support norm computation for the specified `ord`.


.. note::

    On MLX/Metal, `ord=2` for tensors with more than two dimensions is internally passed to 
    `mx.linalg.norm` as `ord=None` (Frobenius norm) to follow MLX's expected signature and avoid CPU fallback. 
    This keeps the behavior aligned with NumPy while staying on GPU.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = lucid.Tensor([3.0, 4.0])
    >>> n = lucid.linalg.norm(a, ord=2)
    >>> print(n)
    Tensor(5.0)

    >>> n1 = lucid.linalg.norm(a, ord=1)
    >>> print(n1)
    Tensor(7.0)
