from typing import TypeAlias
import numpy as np


_Scalar: TypeAlias = int | float
_NumPyArray: TypeAlias = np.ndarray
_ArrayOrScalar: TypeAlias = _Scalar | list[_Scalar] | _NumPyArray

_ShapeLike: TypeAlias = list[int] | tuple[int]
_ArrayLike: TypeAlias = list | _NumPyArray
