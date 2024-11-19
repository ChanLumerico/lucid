lucid.random
============

The `lucid.random` package provides utilities for generating random tensors, 
initializing parameters, and ensuring reproducibility in deep learning experiments. 

These tools are essential for experiments where randomness plays a critical role, 
such as weight initialization and data augmentation.

Overview
--------

This package offers the following functionalities:

- **Random Tensor Generation**: Create tensors with random values following specific distributions.
- **Seed Management**: Ensure reproducibility by controlling the random seed.
- **Specialized Random Sampling**: Generate samples for various applications, 
  such as noise addition or probabilistic modeling.

Key Features
------------

.. rubric:: `Random Tensor Generation`

The package supports a variety of random tensor generation methods, 
including uniform, normal, and custom distributions.

.. admonition:: Example

    Generate random tensors using the `lucid.random` package:

    .. code-block:: python

        >>> import lucid
        >>> random_tensor = lucid.random.randn(3, 3)
        >>> print(random_tensor)

    This generates a 3x3 tensor with random values drawn from a standard normal distribution.

.. tip::

    Use `rand` for uniform distributions and `randint` for discrete random values.

.. rubric:: `Seed Management`

To ensure reproducibility across runs, you can set a global random seed.

.. important::

    Setting a random seed is critical when comparing results from different runs of the same experiment.

.. admonition:: Example

    Set a random seed and generate consistent random values:

    .. code-block:: python

        >>> lucid.random.seed(42)
        >>> consistent_tensor = lucid.random.randn(3, 3)
        >>> print(consistent_tensor)

    Subsequent runs with the same seed will produce identical outputs.

.. caution::

    Remember to set the seed at the beginning of your script to ensure reproducibility 
    across the entire program.

.. rubric:: `Specialized Random Sampling`

The package also includes utilities for generating random integers, sampling with replacement, 
and creating tensors for probabilistic tasks.

.. admonition:: Example

    Generate random integers or perform sampling:

    .. code-block:: python

        >>> random_integers = lucid.random.randint(0, 10, size=(3, 3))
        >>> print(random_integers)

    This creates a 3x3 tensor with random integers between 0 and 10.

Integration with `lucid`
------------------------

The `lucid.random` package is fully compatible with other parts of the `lucid` library. 
Random tensors generated here can be directly used in neural network layers, 
data preprocessing pipelines, or other computational tasks.

.. attention::

    Ensure that your random tensors are correctly shaped and scaled when used as 
    inputs for models or algorithms. Improper initialization may lead to unexpected behavior.

Conclusion
----------

The `lucid.random` package is a versatile utility for generating random data and ensuring 
reproducibility in deep learning workflows. Its seamless integration with the `lucid` 
library makes it a valuable tool for model development and experimentation.

.. admonition:: Learn More
    
    Explore additional functions in the `lucid.random` module by referring to the source 
    code or interactive documentation for further details.