from functools import partial
from types import ModuleType

import numpy as np

from lucid._backend.core import Operation, func_op, _FuncOpReturnType, _GradType
from lucid._backend.metal import mx
from lucid._tensor import Tensor


class embedding_kernel(Operation):
    def __init__(
        self,
        padding_idx: int = -1,
        max_norm: float | None = None,
        norm_type: float = 2.0,
    ) -> None:
        super().__init__()
        self.padding_idx = int(padding_idx)
        self.max_norm = max_norm
        self.norm_type = float(norm_type)
        self._indices = None
        self._num_embeddings = None

    def clear(self) -> None:
        super().clear()
        self.padding_idx = -1
        self.max_norm = None
        self.norm_type = 2.0
        self._indices = None
        self._num_embeddings = None

    @func_op(n_in=2, n_ret=1)
    def cpu(self, indices: Tensor, weight: Tensor) -> _FuncOpReturnType:
        return self._forward(indices, weight, lib_=np)

    @func_op(n_in=2, n_ret=1, device="gpu")
    def gpu(self, indices: Tensor, weight: Tensor) -> _FuncOpReturnType:
        return self._forward(indices, weight, lib_=mx)

    def _forward(
        self, indices: Tensor, weight: Tensor, lib_: ModuleType
    ) -> _FuncOpReturnType:
        idx = indices.data

        if self.max_norm is not None:
            flat = idx.reshape(-1)
            w = weight.data[flat]

            norms = (lib_.abs(w) ** self.norm_type).sum(axis=1) ** (
                1.0 / self.norm_type
            )
            scale = lib_.minimum(1.0, self.max_norm / (norms + (norms == 0)))

            if self.padding_idx >= 0:
                mask = flat == self.padding_idx
                mask_f = mask.astype(scale.dtype)
                scale = scale * (1 - mask_f) + mask_f

            weight.data[flat] = w * scale[:, None]

        out = weight.data[idx]

        self._indices = idx
        self._num_embeddings = int(weight.shape[0])

        self.result = Tensor(out)
        return self.result, partial(self.__grad__, lib_=lib_)

    def __grad__(self, lib_: ModuleType) -> _GradType:
        if self.result is None or self.result.grad is None:
            raise RuntimeError("embedding backward called before forward.")

        if self._indices is None or self._num_embeddings is None:
            raise RuntimeError("embedding cached data missing.")

        grad_out = self.result.grad
        idx = self._indices.reshape(-1)
        grad_flat = grad_out.reshape(idx.shape[0], -1)

        if lib_ is np:
            if self.padding_idx >= 0:
                keep = idx != self.padding_idx
                idx = idx[keep]
                grad_flat = grad_flat[keep]

            grad_w = np.zeros(
                (self._num_embeddings, grad_flat.shape[1]), dtype=grad_out.dtype
            )
            np.add.at(grad_w, idx, grad_flat)

        else:
            grad_w = mx.zeros(
                (self._num_embeddings, grad_flat.shape[1]), dtype=grad_out.dtype
            )
            for i in range(idx.shape[0]):
                if self.padding_idx >= 0 and int(idx[i]) == self.padding_idx:
                    continue
                grad_w = grad_w.at[idx[i]].add(grad_flat[i])

        return None, grad_w
