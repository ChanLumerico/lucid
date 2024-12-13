from typing import Any, Callable, Dict, Sequence
import numpy as np


_Scalar = int | float
_NumPyArray = np.ndarray
_ArrayOrScalar = _Scalar | list[_Scalar] | _NumPyArray

_ShapeLike = list[int] | tuple[int]

_ArrayLike = list | _NumPyArray
_ArrayLikeInt = int | Sequence[int | tuple[int, int]]

_StateDict = Dict[str, Any]

_OptimClosure = Callable[[], Any]
