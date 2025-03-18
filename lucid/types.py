from typing import Any, Callable, Dict, Sequence, Literal

# NOTE: This module remains dependency-free.
import numpy as np
import mlx.core as mx


_DeviceType = Literal["cpu", "gpu"]

_Scalar = int | float
_NumPyArray = np.ndarray
_ArrayOrScalar = _Scalar | list[_Scalar] | _NumPyArray

_ShapeLike = list[int] | tuple[int]

_ArrayLike = list | _NumPyArray
_ArrayLikeInt = int | Sequence[int | tuple[int, int]]

_StateDict = Dict[str, Any]

_OptimClosure = Callable[[], Any]

_EinopsPattern = str


class Numeric:
    def __init__(self, base_dtype: type[_Scalar]) -> None:
        self.base_dtype = base_dtype

    def _dtype_bits(self, dtype: type) -> int:
        if isinstance(dtype, (np.dtype, type)) and hasattr(dtype, "itemsize"):
            return np.dtype(dtype).itemsize * 8

        if isinstance(dtype, (mx.Dtype, type)):
            return dtype.size * 8

        if isinstance(dtype, str):
            try:
                return np.dtype(dtype).itemsize * 8
            except TypeError:
                return dtype.size * 8

        raise TypeError(f"Unsupported dtype: {dtype}")

    def parse(self, tensor_dtype: type, device: _DeviceType) -> type | None:
        bits = self._dtype_bits(tensor_dtype)
        new_dtype = self.base_dtype.__name__ + str(bits)

        return getattr(np if device == "cpu" else mx, new_dtype, None)
