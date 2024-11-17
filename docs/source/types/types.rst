lucid.types
===========

This module defines commonly used type aliases in the `lucid` library. 
These aliases improve code readability and help ensure type consistency across the library.

The type aliases are designed to work seamlessly with Python type hints, providing clarity when working with tensors, arrays, and shapes.

.. automodule:: lucid.types
   :members:
   :undoc-members:

Type Definitions
-----------------

.. currentmodule:: lucid.types

.. rubric:: `_Scalar`

**Definition**:  
Represents a single scalar value, which can be either an integer or a float.  
This is commonly used for operations that involve scalar constants or parameters.

Example:

.. code-block:: python

    scalar: _Scalar = 3.14

.. rubric:: `_NumPyArray`

**Definition**:  
Represents a NumPy array (`np.ndarray`), which is a primary data structure in numerical computing.

Example:

.. code-block:: python

    arr: _NumPyArray = np.array([1, 2, 3])

.. rubric:: `_ArrayOrScalar`

**Definition**:  
Represents a flexible type that can be one of the following:

- A single scalar (`_Scalar`).
- A list of scalars (`list[_Scalar]`).
- A NumPy array (`_NumPyArray`).

Example:

.. code-block:: python

    def process(data: _ArrayOrScalar):
        pass

    process(42)                          # Single scalar
    process([1.2, 3.4, 5.6])             # List of scalars
    process(np.array([7, 8, 9]))         # NumPy array

.. rubric:: `_ShapeLike`

**Definition**:  
Represents a shape, which can be a list or tuple of integers.  
This is typically used for specifying the dimensions of tensors or arrays.

Example:

.. code-block:: python

    shape: _ShapeLike = (2, 3)

.. rubric:: `_ArrayLike`

**Definition**:  
Represents data that can be interpreted as an array.  
This includes Python lists and NumPy arrays.

Example:

.. code-block:: python

    data: _ArrayLike = [1, 2, 3]
    array: np.ndarray = np.asarray(data)

Admonitions and Usage Guidelines
---------------------------------

.. admonition:: Why Use Type Aliases?

    Type aliases provide the following benefits:
    
    - They improve code readability by using meaningful names instead of raw type definitions.
    - They ensure consistency across the library, reducing the risk of errors.
    - They help document the intent and expected input/output of functions clearly.

.. note::

    These type aliases are especially useful in libraries like `lucid`, where tensors, arrays, and their shapes play a central role.

Example Use Cases
-----------------

Here are some examples demonstrating how these type aliases can be used in function definitions:

.. code-block:: python

    from lucid.types import _Scalar, _ArrayOrScalar, _ShapeLike

    def scale(data: _ArrayOrScalar, factor: _Scalar) -> _ArrayOrScalar:
        """Scales the input data by a scalar factor."""
        pass

    def reshape(array: _NumPyArray, shape: _ShapeLike) -> _NumPyArray:
        """Reshapes a NumPy array to the specified shape."""
        return np.reshape(array, shape)
