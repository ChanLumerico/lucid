import functools
import math
from types import ModuleType

import numpy as np

from lucid._backend.core import Operation, func_op, _FuncOpReturnType, _GradType
from lucid._backend.metal import mx
from lucid._tensor import Tensor

from lucid.types import _DeviceType, _TensorData


def _make_causal_mask(lib_: ModuleType, L: int, S: int, dtype: object) -> _TensorData:
    triu = getattr(lib_, "triu", None)
    ones = getattr(lib_, "ones", None)
    if triu is None or ones is None:
        mask = np.triu(np.ones((L, S), dtype=np.float32), k=1)
        if lib_ is mx:
            mask = mx.array(mask)
    else:
        mask = triu(ones((L, S), dtype=dtype), k=1)
    return mask * (-1e12)


class scaled_dot_product_attention_kernel(Operation):
    def __init__(
        self,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        self.attn_mask = attn_mask
        self.is_causal = bool(is_causal)
        self.scale = scale

        self._q = None
        self._k = None
        self._v = None
        self._attn = None
        self._scale = None

    def clear(self) -> None:
        super().clear()
        self._q = None
        self._k = None
        self._v = None
        self._attn = None
        self._scale = None

    @func_op(n_in=3, n_ret=1)
    def cpu(self, q: Tensor, k: Tensor, v: Tensor) -> _FuncOpReturnType:
        return self._forward(q, k, v, lib_=np, device="cpu")

    @func_op(n_in=3, n_ret=1, device="gpu")
    def gpu(self, q: Tensor, k: Tensor, v: Tensor) -> _FuncOpReturnType:
        return self._forward(q, k, v, lib_=mx, device="gpu")

    def _forward(
        self, q: Tensor, k: Tensor, v: Tensor, lib_: ModuleType, device: _DeviceType
    ) -> _FuncOpReturnType:
        qd = q.data
        kd = k.data
        vd = v.data

        scale = self.scale
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])

        kt = lib_.swapaxes(kd, -1, -2)
        scores = lib_.matmul(qd, kt) * scale

        if self.is_causal:
            L = q.shape[-2]
            S = k.shape[-2]
            scores = scores + _make_causal_mask(lib_, L, S, dtype=scores.dtype)

        if self.attn_mask is not None:
            scores = scores + self.attn_mask.data

        max_val = lib_.max(scores, axis=-1, keepdims=True)
        exp_x = lib_.exp(scores - max_val)
        sum_exp = lib_.sum(exp_x, axis=-1, keepdims=True)
        attn = exp_x / sum_exp

        out = lib_.matmul(attn, vd)

        self._q = qd
        self._k = kd
        self._v = vd
        self._attn = attn
        self._scale = scale

        self.result = Tensor(out, device=device)
        return self.result, functools.partial(self.__grad__, lib_=lib_)

    def __grad__(self, lib_: ModuleType) -> _GradType:
        if self.result is None or self.result.grad is None:
            raise RuntimeError("attention backward called before forward.")

        if self._attn is None or self._q is None or self._k is None or self._v is None:
            raise RuntimeError("attention cached data missing.")

        dy = self.result.grad
        attn = self._attn
        qd = self._q
        kd = self._k
        vd = self._v
        scale = self._scale if self._scale is not None else 1.0

        attn_t = lib_.swapaxes(attn, -1, -2)
        dV = lib_.matmul(attn_t, dy)

        v_t = lib_.swapaxes(vd, -1, -2)
        dA = lib_.matmul(dy, v_t)

        dot = lib_.sum(dA * attn, axis=-1, keepdims=True)
        dS = attn * (dA - dot)

        dS = dS * scale
        dQ = lib_.matmul(dS, kd)
        dK = lib_.matmul(lib_.swapaxes(dS, -1, -2), qd)

        return dQ, dK, dV
