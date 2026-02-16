lucid.linalg.matrix_power
=========================

.. autofunction:: lucid.linalg.matrix_power

The `matrix_power` function computes the integer power of a square matrix. 
Given a square matrix **A** and an integer exponent **n**, it returns **A** 
raised to the power **n**, denoted as :math:`\mathbf{A}^n`.

Function Signature
------------------

.. code-block:: python

    def matrix_power(a: Tensor, n: int) -> Tensor

Parameters
----------

- **a** (*Tensor*):
  The input tensor, which must be a square matrix (a two-dimensional tensor with equal dimensions).

- **n** (*int*):
  The exponent to which the matrix is to be raised. Can be any integer (positive, zero, or negative).

Returns
-------

- **Tensor**: The resulting tensor after raising the input matrix **a** to the power **n**.

Forward Calculation
-------------------

The forward calculation for `matrix_power` computes the matrix power of **A** based on the exponent **n**:

- **For** :math:`n > 0`:
  
  The function computes the matrix multiplied by itself :math:`n` times:

  .. math::

      \mathbf{A}^n = \underbrace{\mathbf{A} \times \mathbf{A} 
      \times \dots \times \mathbf{A}}_{n \text{ times}}

- **For** :math:`n = 0`:

  The function returns the identity matrix of the same dimension as **A**:

  .. math::

      \mathbf{A}^0 = \mathbf{I}

- **For** :math:`n < 0`:

  The function computes the inverse of **A** raised to the absolute value of **n**:

  .. math::

      \mathbf{A}^n = \left( \mathbf{A}^{-1} \right)^{|n|}

Backward Gradient Calculation
-----------------------------

The gradient of the matrix power with respect to the input matrix **A** is 
computed differently based on the value of **n**:

- **For** :math:`n \ne 0`:

  The gradient is calculated using the chain rule applied to matrix multiplication:

  .. math::

      \frac{\partial \mathbf{A}^n}{\partial \mathbf{A}} = 
      \sum_{k=0}^{|n|-1} \mathbf{A}^{n - s(k + 1)} \cdot 
      \frac{\partial \mathbf{A}^n}{\partial \mathbf{A}^n} \cdot \mathbf{A}^{s k}

  where:

  - :math:`s = \operatorname{sign}(n)` (i.e., :math:`s = 1` if :math:`n > 0`, :math:`s = -1` if :math:`n < 0`).
  - :math:`\frac{\partial \mathbf{A}^n}{\partial \mathbf{A}^n}` 
    is the gradient of the output with respect to itself (often denoted as **grad_output** in code).

  **Explanation**:

  - The summation accounts for each occurrence of **A** in the sequence of multiplications.
  - For negative exponents, the gradient includes an additional negative sign due to the properties of matrix inversion.

- **For** :math:`n = 0`:

  Since the output is the identity matrix, which does not depend on **A**, the gradient with respect to **A** is zero:

  .. math::

      \frac{\partial \mathbf{A}^0}{\partial \mathbf{A}} = \mathbf{0}

These gradients are essential for backpropagation in optimization algorithms, 
enabling models that use matrix powers to learn from data.

Raises
------

.. attention::

  - **ValueError**: If the input tensor **a** is not a square matrix 
    (i.e., it is not two-dimensional or its dimensions are not equal).

  - **LinAlgError**: If the matrix inverse cannot be computed when :math:`n < 0` 
    (e.g., if **a** is singular or not invertible).

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = lucid.Tensor([[2.0, 0.0], [0.0, 2.0]])
    >>> result = lucid.linalg.matrix_power(a, 3)
    >>> print(result)
    Tensor([[8.0, 0.0],
            [0.0, 8.0]])

    >>> result = lucid.linalg.matrix_power(a, -1)
    >>> print(result)
    Tensor([[0.5, 0.0],
            [0.0, 0.5]])

    >>> result = lucid.linalg.matrix_power(a, 0)
    >>> print(result)
    Tensor([[1.0, 0.0],
            [0.0, 1.0]])

.. note::

  - The input tensor **a** must be a square matrix with shape :math:`(n, n)`.
  - The exponent **n** can be any integer, including zero and negative integers.
  - For negative exponents, the function computes the inverse of **a** before raising it to the power :math:`|n|`.
  - If **a** is not invertible (i.e., singular), a **LinAlgError** will be raised when **n** is negative.
  - The function does not support non-integer exponents.
  - This function does not support batch processing; each input must be a single square matrix.
