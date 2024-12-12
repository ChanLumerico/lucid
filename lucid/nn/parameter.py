import numpy as np

from lucid._tensor import Tensor
from lucid.types import _ArrayOrScalar


__all__ = ["Parameter", "Buffer"]


class Parameter(Tensor):
    def __init__(self, data: Tensor | _ArrayOrScalar, dtype=np.float32) -> None:
        if isinstance(data, Tensor):
            data = data.data
        super().__init__(data, requires_grad=True, keep_grad=True, dtype=dtype)


class Buffer(Tensor):
    def __init__(self, data: Tensor | _ArrayOrScalar, dtype=np.float32) -> None:
        if isinstance(data, Tensor):
            data = data.data
        super().__init__(data, requires_grad=False, keep_grad=False, dtype=dtype)
