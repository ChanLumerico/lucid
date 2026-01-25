import functools
from types import ModuleType

import numpy as np

from lucid._backend.core import Operation, func_op, _FuncOpReturnType, _GradType
from lucid._backend.metal import mx
from lucid._tensor import Tensor

from lucid.types import _DeviceType, _TensorData


def _as_int_array(arr, lib_: ModuleType) -> _TensorData:
    if lib_ is np:
        return arr.astype(np.int64)
    return arr.astype(mx.int32)


class embedding_kernel(Operation):
    def __init__(self) -> None:
        super().__init__()
        self._indices = None
        self._num_embeddings = None

    def clear(self) -> None:
        super().clear()
        self._indices = None
        self._num_embeddings = None

    @func_op(n_in=2, n_ret=1)
    def cpu(self, indices: Tensor, weight: Tensor) -> _FuncOpReturnType:
        return self._forward(indices, weight, lib_=np, device="cpu")

    @func_op(n_in=2, n_ret=1, device="gpu")
    def gpu(self, indices: Tensor, weight: Tensor) -> _FuncOpReturnType:
        return self._forward(indices, weight, lib_=mx, device="gpu")

    def _forward(
        self, indices: Tensor, weight: Tensor, lib_: ModuleType, device: _DeviceType
    ) -> _FuncOpReturnType:
        idx = _as_int_array(indices.data, lib_)
        out = weight.data[idx]

        self._indices = idx
        self._num_embeddings = int(weight.shape[0])

        self.result = Tensor(out, device=device)
        return self.result, functools.partial(self.__grad__, lib_=lib_)

    def __grad__(self, lib_: ModuleType) -> _GradType:
        if self.result is None or self.result.grad is None:
            raise RuntimeError("embedding backward called before forward.")
        if self._indices is None or self._num_embeddings is None:
            raise RuntimeError("embedding cached data missing.")

        grad_out = self.result.grad
        idx = self._indices.reshape(-1)
        grad_flat = grad_out.reshape(idx.shape[0], -1)

        if lib_ is np:
            grad_w = np.zeros(
                (self._num_embeddings, grad_flat.shape[1]), dtype=grad_out.dtype
            )
            np.add.at(grad_w, idx, grad_flat)
        else:
            grad_w = mx.zeros(
                (self._num_embeddings, grad_flat.shape[1]), dtype=grad_out.dtype
            )
            for i in range(idx.shape[0]):
                grad_w = grad_w.at[idx[i]].add(grad_flat[i])

        return None, grad_w
