lucid.newaxis
=============

The `newaxis` is used to add a new dimension to a tensor or array, 
effectively increasing its dimensionality by one. 

This is commonly used in machine learning and scientific computing for 
reshaping data and ensuring compatibility with broadcasting rules.

Usage
-----

The `newaxis` can be used in indexing to create a new axis in a tensor's shape. 

This is especially useful for aligning tensors with different shapes for 
element-wise operations or reshaping data.

.. admonition:: Example 1: Adding a new axis to a 1D tensor
    
    In this example, `newaxis` is used to add a new axis at the beginning of a 1D tensor.

    .. code-block:: python

        >>> import lucid
        >>> a = Tensor([1, 2, 3])
        >>> print(a.shape)  # Output: (3,)
        >>> a = a[None, :]  # Adds a new axis at the beginning
        >>> print(a.shape)  # Output: (1, 3)

    The shape of the tensor `a` is transformed from `(3,)` to `(1, 3)` by adding a new axis.

.. admonition:: Example 2: Adding a new axis to a 2D tensor
    
    Here, we add a new axis in the middle of a 2D tensor.

    .. code-block:: python

        >>> b = Tensor([[1, 2], [3, 4]])
        >>> print(b.shape)  # Output: (2, 2)
        >>> b = b[:, None, :]  # Adds a new axis in the second position
        >>> print(b.shape)  # Output: (2, 1, 2)

    In this case, the shape of the tensor `b` changes from `(2, 2)` to `(2, 1, 2)`.

.. admonition:: Example 3: Adding a new axis to align tensors for broadcasting
    
    Sometimes, `newaxis` is used to align tensors for broadcasting in operations 
    like addition or multiplication.

    .. code-block:: python
        
        >>> c = Tensor([1, 2, 3])
        >>> d = Tensor([[4], [5], [6]])
        >>> print((c + d).shape)  # Output: (3, 3)

    By adding a new axis to `c`, it aligns with the shape of `d`, 
    allowing the element-wise addition to work.

Conclusion
----------

The `newaxis` is a simple yet powerful tool for reshaping tensors by adding new axes. 

It is especially helpful for aligning tensors to make them compatible for broadcasting 
and for preparing data in machine learning tasks.

