from typing import Any, Callable, Dict, Sequence
import numpy as np

_base_dtype: np.floating = np.float32

_Scalar = int | float
_NumPyArray = np.ndarray
_ArrayOrScalar = _Scalar | list[_Scalar] | _NumPyArray

_ShapeLike = list[int] | tuple[int]

_ArrayLike = list | _NumPyArray
_ArrayLikeInt = int | Sequence[int | tuple[int, int]]

_StateDict = Dict[str, Any]

_OptimClosure = Callable[[], Any]


def _change_base_dtype(dtype: type) -> None:
    _base_dtype = dtype
