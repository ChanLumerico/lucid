lucid.einops
============

The `lucid.einops` module provides powerful tensor manipulation capabilities 
for the `lucid` framework, inspired by the `einops` library. 
It offers flexible and intuitive operations for rearranging and reducing tensor 
dimensions, facilitating efficient deep learning model development.

Overview
--------

The `lucid.einops` package enables tensor transformations that go beyond 
traditional reshaping, providing:

- **Rearrange**: Change the order of dimensions in a flexible manner.
- **Reduce**: Aggregate tensor values along specified axes using various reduction methods.
- **Repeat**: Expand tensor elements along specified dimensions.

These operations make it easy to manipulate tensor shapes for neural networks, 
enabling compatibility with various architectures and optimizing computational efficiency.

.. note::

    `lucid.einops` provides an expressive way to handle tensor manipulations 
    without manual indexing, making code more readable and concise.

`rearrange`
^^^^^^^^^^^

The `rearrange` function allows for flexible reordering and reshaping of 
tensors using a notation similar to Einstein summation.

.. code-block:: python

    def rearrange(tensor: Tensor, pattern: str, **dimensions) -> Tensor

- **tensor** (*Tensor*): Input tensor.
- **pattern** (*str*): A string describing the desired rearrangement.
- **dimensions** (*dict*): Optional named dimensions for expanding or collapsing axes.

.. admonition:: Example

    .. code-block:: python

        >>> import lucid.einops as einops
        >>> t = lucid.Tensor([[1, 2], [3, 4]])
        >>> out = einops.rearrange(t, 'h w -> (h w)')
        >>> print(out)
        Tensor([1, 2, 3, 4])

.. warning::

    Be cautious when collapsing dimensions, as improper reshaping may lead to 
    unexpected gradient behaviors in backpropagation.

`reduce`
^^^^^^^^

The `reduce` function performs reduction operations along specified axes, 
such as summation, mean, or max pooling.

.. code-block:: python

    def reduce(tensor: Tensor, pattern: str, reduction: str, **dimensions) -> Tensor

- **tensor** (*Tensor*): Input tensor.
- **pattern** (*str*): A string defining the reduction pattern.
- **reduction** (*str*): The reduction operation (e.g., 'sum', 'mean', 'max').
- **dimensions** (*dict*): Optional named dimensions.

.. admonition:: Example

    .. code-block:: python

        >>> import lucid.einops as einops
        >>> t = lucid.Tensor([[1, 2], [3, 4]])
        >>> out = einops.reduce(t, 'h w -> h', reduction='sum')
        >>> print(out)
        Tensor([3, 7])

.. important::

    Ensure that the specified reduction operation (`sum`, `mean`, `max`, etc.) 
    aligns with your intended data aggregation.

`repeat`
^^^^^^^^

The `repeat` function enables element-wise expansion along specified dimensions, 
allowing for controlled duplication of tensor elements.

.. code-block:: python

    def repeat(tensor: Tensor, pattern: str, **dimensions) -> Tensor

- **tensor** (*Tensor*): Input tensor.
- **pattern** (*str*): A string defining the repetition pattern.
- **dimensions** (*dict*): Optional named dimensions specifying expansion sizes.

.. admonition:: Example

    .. code-block:: python

        >>> import lucid.einops as einops
        >>> t = lucid.Tensor([1, 2, 3])
        >>> out = einops.repeat(t, 'i -> i j', j=2)
        >>> print(out)
        Tensor([[1, 1],
                [2, 2],
                [3, 3]])

.. warning::

    Ensure that the total number of elements before and after repetition matches.
    Mismatched sizes will result in an error.

Advantages
----------

- **Intuitive Syntax**: Allows concise, readable tensor operations.
- **Optimized Performance**: Reduces unnecessary reshaping and copying of tensors.
- **Flexible Dimensionality Handling**: Works seamlessly across different input shapes.

.. tip::

    Use `rearrange`, `reduce`, and `repeat` to make neural network architectures 
    more adaptable and efficient!

.. caution::

    Always verify the output shape after applying transformations to 
    prevent unintended dimensional mismatches in neural network layers.

Conclusion
----------

The `lucid.einops` module brings the power of `einops`-style tensor 
manipulation into `lucid`, offering a structured approach to handling tensor 
transformations crucial for deep learning applications.

.. hint::

    When unsure about reshaping patterns, start by breaking them into smaller, 
    interpretable steps before applying complex transformations.
