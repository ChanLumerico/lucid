lucid.random.seed
==================

.. autofunction:: lucid.random.seed

The `seed` function initializes the random number generator with a specified seed value, 
ensuring reproducibility of random operations in the `lucid.random` package.

Function Signature
------------------

.. code-block:: python

    def seed(seed: int) -> None

Parameters
----------

- **seed** (*int*): The seed value for the random number generator. 
  Providing the same seed guarantees consistent random values across different runs.

Returns
-------

- **None**: This function does not return any value. 
  It directly modifies the state of the random number generator.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> lucid.random.seed(123)
    >>> print(lucid.random.randn(3))
    [ 0.99734545 -0.90709145 -0.57366333]

    >>> lucid.random.seed(123)
    >>> print(lucid.random.randn(3))
    [ 0.99734545 -0.90709145 -0.57366333]

As shown above, resetting the seed produces identical random numbers.

.. note::

    - This function is essential for debugging and reproducibility, especially in machine learning experiments.
    - For global consistency, ensure the seed is set before performing any random operation.