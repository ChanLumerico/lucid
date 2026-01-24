import functools
from types import ModuleType

import numpy as np

from lucid._backend.core import Operation, func_op, _FuncOpReturnType, _GradType
from lucid._backend.metal import mx
from lucid._tensor import Tensor
from lucid.types import _DeviceType


def _norm_axis(axis: int, ndim: int) -> int:
    return axis if axis >= 0 else axis + ndim


class softmax_kernel(Operation):
    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis
        self._axis = None
        self._y = None

    def clear(self) -> None:
        super().clear()
        self._axis = None
        self._y = None

    @func_op(n_in=1, n_ret=1, device="cpu")
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        return self._forward(a, lib_=np, device="cpu")

    @func_op(n_in=1, n_ret=1, device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        return self._forward(a, lib_=mx, device="gpu")

    def _forward(
        self, a: Tensor, lib_: ModuleType, device: _DeviceType
    ) -> _FuncOpReturnType:
        axis = _norm_axis(self.axis, a.ndim)
        max_val = lib_.max(a.data, axis=axis, keepdims=True)
        exp_x = lib_.exp(a.data - max_val)
        sum_exp = lib_.sum(exp_x, axis=axis, keepdims=True)
        y = exp_x / sum_exp

        self._axis = axis
        self._y = y

        self.result = Tensor(y, device=device)
        return self.result, functools.partial(self.__grad__, lib_=lib_)

    def __grad__(self, lib_: ModuleType) -> _GradType:
        if self.result is None or self.result.grad is None:
            raise RuntimeError("softmax backward called before forward.")
        if self._y is None or self._axis is None:
            raise RuntimeError("softmax cached data missing.")

        dy = self.result.grad
        y = self._y
        axis = self._axis

        dot = lib_.sum(dy * y, axis=axis, keepdims=True)
        dx = y * (dy - dot)
        return dx


class sigmoid_kernel(Operation):
    def __init__(self) -> None:
        super().__init__()
        self._y = None

    def clear(self) -> None:
        super().clear()
        self._y = None

    @func_op(n_in=1, n_ret=1, device="cpu")
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        return self._forward(a, lib_=np, device="cpu")

    @func_op(n_in=1, n_ret=1, device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        return self._forward(a, lib_=mx, device="gpu")

    def _forward(
        self, a: Tensor, lib_: ModuleType, device: _DeviceType
    ) -> _FuncOpReturnType:
        y = 1.0 / (1.0 + lib_.exp(-a.data))
        self._y = y
        self.result = Tensor(y, device=device)
        return self.result, functools.partial(self.__grad__)

    def __grad__(self) -> _GradType:
        if self.result is None or self.result.grad is None or self._y is None:
            raise RuntimeError("sigmoid backward called before forward.")

        dy = self.result.grad
        y = self._y

        dx = dy * y * (1 - y)
        return dx


class gelu_kernel(Operation):
    def __init__(self) -> None:
        super().__init__()
        self._x = None

    def clear(self) -> None:
        super().clear()
        self._x = None

    @func_op(n_in=1, n_ret=1, device="cpu")
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        return self._forward(a, lib_=np, device="cpu")

    @func_op(n_in=1, n_ret=1, device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        return self._forward(a, lib_=mx, device="gpu")

    def _forward(
        self, a: Tensor, lib_: ModuleType, device: _DeviceType
    ) -> _FuncOpReturnType:
        self._x = a.data
        c = lib_.sqrt(2.0 / lib_.pi)
        y = 0.5 * a.data * (1.0 + lib_.tanh(c * (a.data + 0.044715 * (a.data**3))))

        self.result = Tensor(y, device=device)
        return self.result, functools.partial(self.__grad__, lib_=lib_)

    def __grad__(self, lib_: ModuleType) -> _GradType:
        if self.result is None or self.result.grad is None or self._x is None:
            raise RuntimeError("gelu backward called before forward.")

        x = self._x
        dy = self.result.grad
        c = lib_.sqrt(2.0 / lib_.pi)
        t = c * (x + 0.044715 * x**3)
        dt = c * (1 + 3 * 0.044715 * x**2)
        sech2 = 1.0 / lib_.cosh(t) ** 2

        dx = 0.5 * (1 + lib_.tanh(t)) + 0.5 * x * sech2 * dt
        return dy * dx


class silu_kernel(Operation):
    def __init__(self) -> None:
        super().__init__()
        self._x = None
        self._sig = None

    def clear(self) -> None:
        super().clear()
        self._x = None
        self._sig = None

    @func_op(n_in=1, n_ret=1, device="cpu")
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        return self._forward(a, lib_=np, device="cpu")

    @func_op(n_in=1, n_ret=1, device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        return self._forward(a, lib_=mx, device="gpu")

    def _forward(
        self, a: Tensor, lib_: ModuleType, device: _DeviceType
    ) -> _FuncOpReturnType:
        self._x = a.data
        sig = 1.0 / (1.0 + lib_.exp(-a.data))
        self._sig = sig
        y = a.data * sig

        self.result = Tensor(y, device=device)
        return self.result, functools.partial(self.__grad__)

    def __grad__(self) -> _GradType:
        if (
            self.result is None
            or self.result.grad is None
            or self._x is None
            or self._sig is None
        ):
            raise RuntimeError("silu backward called before forward.")

        dy = self.result.grad
        sig = self._sig
        x = self._x

        dx = dy * (sig + x * sig * (1 - sig))
        return dx
