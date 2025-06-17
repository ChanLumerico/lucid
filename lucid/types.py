from typing import Any, Callable, Dict, Sequence, Literal, TypeAlias
import re

# NOTE: This module remains module independency.
import numpy as np
import mlx.core as mx


_DeviceType = Literal["cpu", "gpu"]

_Scalar = int | float | complex
_NumPyArray: TypeAlias = np.ndarray
_MLXArray: TypeAlias = mx.array

_Gradient = _NumPyArray | _MLXArray | None

_ArrayOrScalar = _Scalar | list[_Scalar] | _NumPyArray | _MLXArray

_BuiltinNumeric = type[bool | int | float | complex]

_ShapeLike = list[int] | tuple[int]

_ArrayLike = list | _NumPyArray | _MLXArray
_ArrayLikeInt = int | Sequence[int | tuple[int, int]]

_StateDict = Dict[str, Any]

_OptimClosure = Callable[[], Any]

_EinopsPattern = str


class Numeric:
    def __init__(
        self, base_dtype: type[int | float | complex], bits: int | None
    ) -> None:
        self.base_dtype = base_dtype
        self.base_str = base_dtype.__name__
        self.bits = bits

        self._np_dtype: type | None = None
        self._mlx_dtype: type | None = None

        if bits is not None:
            self._np_dtype = getattr(np, self.base_str + str(bits))
            bits_mlx = bits
            if (
                mx.default_device().type is mx.DeviceType.gpu
                and self.base_dtype is float
                and bits == 64
            ):
                bits_mlx = 32
            self._mlx_dtype = getattr(mx, self.base_str + str(bits_mlx))

    @property
    def cpu(self) -> type | None:
        return self._np_dtype

    @property
    def gpu(self) -> type | None:
        return self._mlx_dtype

    @property
    def is_bit_free(self) -> bool:
        return self.bits is None

    def parse(self, device: _DeviceType) -> type | None:
        if device == "cpu":
            return self.cpu
        else:
            return self.gpu

    def auto_parse(self, data_dtype: type, device: _DeviceType) -> type | None:
        bits = self._dtype_bits(data_dtype)
        new_dtype = self.base_dtype.__name__ + str(bits)

        return getattr(np if device == "cpu" else mx, new_dtype, None)

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

    def __str__(self) -> str:
        return self.base_dtype.__name__ + str(self.bits)

    def __repr__(self) -> str:
        return (
            f"(base_dtype={self.base_dtype.__name__}, bits={self.bits},"
            + f" _np_dtype={self.cpu}, _mlx_dtype={self.gpu})",
        )[0]


Int = Numeric(int, None)
Int8 = Numeric(int, bits=8)
Int16 = Numeric(int, bits=16)
Int32 = Numeric(int, bits=32)
Int64 = Numeric(int, bits=64)

Float = Numeric(float, None)
Float16 = Numeric(float, bits=16)
Float32 = Numeric(float, bits=32)
Float64 = Numeric(float, bits=64)

Complex = Numeric(complex, None)
Complex64 = Numeric(complex, bits=64)

numeric_dict = {
    "int": {"8": Int8, "16": Int16, "32": Int32, "64": Int64},
    "float": {"16": Float16, "32": Float32, "64": Float64},
    "complex": {"64": Complex64},
}


def to_numeric_type(data_dtype: type) -> Numeric:
    str_dtype = str(data_dtype).split(".")[-1]

    name = re.findall(r"[a-z]+", str_dtype)[0]
    bits = re.findall(r"\d+", str_dtype)[0]

    return numeric_dict[name][bits]
