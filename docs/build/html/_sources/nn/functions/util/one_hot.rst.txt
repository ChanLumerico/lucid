nn.functional.one_hot
=====================

.. autofunction:: lucid.nn.functional.one_hot

The `one_hot` function converts a tensor of integer indices into a one-hot encoded tensor, 
adding a new last dimension of size `num_classes`. This is commonly used in classification 
problems where categorical labels need to be converted to a binary format.

Function Signature
------------------

.. code-block:: python

    def one_hot(
        input_: Tensor,
        num_classes: int = -1,
        dtype: Numeric | bool | None = None
    ) -> Tensor

Parameters
----------
- **input_** (*Tensor*):
  A tensor of integers representing class indices. Must contain only non-negative 
  integer values.

- **num_classes** (*int*, optional):
  The total number of classes. If set to -1 (default), the number of classes is 
  inferred from the maximum index in the input (i.e., `input_.max() + 1`).

- **dtype** (*Numeric | bool | None*, optional):
  The desired output data type. If None (default), the one-hot output will 
  use the default floating type.

Returns
-------
- **Tensor**:
  A tensor of shape `(*input_.shape, num_classes)`, where each index in the 
  input is converted into a one-hot encoded vector.

One-Hot Encoding Logic
----------------------

For each index in `input_`, the function sets the corresponding position in 
the last dimension to `1`:

.. math::

    \text{output}[i_1, i_2, \dots, i_n, c] =
    \begin{cases}
        1, & \text{if } c = \text{input_}[i_1, i_2, \dots, i_n] \\
        0, & \text{otherwise}
    \end{cases}

If `num_classes` is not specified, it is inferred as the maximum value in `input_` plus one.

Examples
--------

**Basic one-hot encoding:**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn.functional as F
    >>> input_ = lucid.Tensor([[0, 2], [1, 3]], dtype=int)
    >>> output = F.one_hot(input_, num_classes=4)
    >>> print(output.shape)
    (2, 2, 4)

    >>> print(output)
    Tensor([
        [[1, 0, 0, 0],
         [0, 0, 1, 0]],
        [[0, 1, 0, 0],
         [0, 0, 0, 1]]
    ])

**Inferring `num_classes`:**

.. code-block:: python

    >>> input_ = lucid.Tensor([0, 1, 2])
    >>> output = F.one_hot(input_)
    >>> print(output.shape)
    (3, 3)

    >>> print(output)
    Tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

**Specifying a custom dtype:**

.. code-block:: python

    >>> output = F.one_hot(input_, num_classes=4, dtype=lucid.Bool)
    >>> print(output.dtype)
    bool

.. note::

    - The input tensor must contain integers only.
    - The returned tensor will have one more dimension than the input: 
      the last dimension will be `num_classes`.

    - If `input_` contains a value greater than or equal to `num_classes`, 
      an error will be raised.
