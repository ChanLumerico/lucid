lucid.Numeric
=============

.. autoclass:: lucid.types.Numeric

The `Numeric` class in Lucid represents an abstract numeric type, 
encoding both Python's built-in numeric types (e.g., `int`, `float`, `complex`), 
and their corresponding NumPy and MLX (Metal for Apple Silicon) types.

It enables unified handling of dtypes across different devices (CPU/GPU) and platforms.

Class Signature
---------------

.. code-block:: python

    class Numeric:
        def __init__(
            base_dtype: type[int | float | complex],
            bits: int | None
        ) -> None

Parameters
----------
- **base_dtype** (*type[int | float | complex]*):
    The base numeric type category.

- **bits** (*int | None*):
    The bit precision of the numeric type. If `None`, it's considered a bit-free type.

Attributes
----------
- **base_dtype** (*type*):
    The built-in Python type (`int`, `float`, or `complex`).

- **bits** (*int | None*):
    Number of bits (e.g., 32 for float32). `None` for dynamic or unspecified bit-width.

- **cpu** (*type | None*):
    NumPy dtype corresponding to this `Numeric` object (e.g., `np.float32`).

- **gpu** (*type | None*):
    MLX dtype corresponding to this `Numeric` object (e.g., `mx.float32`).

- **is_bit_free** (*bool*):
    True if `bits` is `None`, indicating dynamic typing.

Methods
-------
- **parse(device: Literal["cpu", "gpu"]) -> type | None**:
    Returns the appropriate dtype (NumPy or MLX) based on the specified device.

- **auto_parse(data_dtype: type, device: Literal["cpu", "gpu"]) -> type | None**:
    Infers and returns a dtype with the same bit-width as the given `data_dtype`.

- **_dtype_bits(dtype: type) -> int**:
    Extracts bit-width from a NumPy, MLX, or string dtype.

Representation
--------------

.. code-block:: python

    >>> Float32 = Numeric(float, 32)
    >>> print(Float32)
    float32
    >>> print(repr(Float32))
    (base_dtype=float, bits=32, _np_dtype=<class 'numpy.float32'>, _mlx_dtype=<class 'mlx.core.float32'>)

.. note::
   A `Numeric` instance encapsulates the corresponding types across Python, 
   NumPy, and MLX in a device-agnostic manner.

Usage Examples
--------------

.. code-block:: python

    >>> from lucid.types import Numeric
    >>> dtype = Numeric(float, 64)
    >>> dtype.base_dtype
    <class 'float'>
    >>> dtype.cpu
    <class 'numpy.float64'>
    >>> dtype.gpu
    <class 'mlx.core.float64'>
    >>> dtype.parse("gpu")
    <class 'mlx.core.float64'>

    >>> # Using auto_parse
    >>> dtype.auto_parse(np.float32, "gpu")
    <class 'mlx.core.float32'>

.. tip::
   Use predefined instances such as `lucid.Float32`, `lucid.Int64`, `lucid.Complex64` 
   for common configurations.

.. warning::
   If `bits` is `None`, MLX/NumPy dtypes will be resolved dynamically based on the input data.

