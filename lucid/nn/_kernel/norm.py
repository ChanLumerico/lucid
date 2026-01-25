import functools
from types import ModuleType
from typing import Sequence

import numpy as np

from lucid._backend.core import Operation, func_op, _FuncOpReturnType, _GradType
from lucid._backend.metal import mx

from lucid._tensor import Tensor
from lucid.types import _DeviceType


def _norm_axes(ndim: int, normalized_shape: Sequence[int]) -> tuple[int, ...]:
    return tuple(range(ndim - len(normalized_shape), ndim))


def _broadcast_shape(ndim: int, normalized_shape: Sequence[int]) -> tuple[int, ...]:
    return (1,) * (ndim - len(normalized_shape)) + tuple(normalized_shape)


class layer_norm_kernel(Operation):
    def __init__(
        self,
        normalized_shape: Sequence[int],
        eps: float = 1e-5,
        has_weight: bool = True,
        has_bias: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = tuple(int(v) for v in normalized_shape)
        self.eps = float(eps)
        self.has_weight = bool(has_weight)
        self.has_bias = bool(has_bias)

        self._xhat = None
        self._rstd = None
        self._norm_axes = None
        self._n = None

    def clear(self) -> None:
        super().clear()
        self._xhat = None
        self._rstd = None
        self._norm_axes = None
        self._n = None

    @func_op(n_in=3, n_ret=1, device="cpu")
    def cpu(self, a: Tensor, w: Tensor, b: Tensor) -> _FuncOpReturnType:
        return self._forward(a, w, b, lib_=np, device="cpu")

    @func_op(n_in=3, n_ret=1, device="gpu")
    def gpu(self, a: Tensor, w: Tensor, b: Tensor) -> _FuncOpReturnType:
        return self._forward(a, w, b, lib_=mx, device="gpu")

    def _forward(
        self,
        a: Tensor,
        w: Tensor,
        b: Tensor,
        lib_: ModuleType,
        device: _DeviceType,
    ) -> _FuncOpReturnType:
        norm_axes = _norm_axes(a.ndim, self.normalized_shape)
        n = int(np.prod(self.normalized_shape))
        mean = lib_.mean(a.data, axis=norm_axes, keepdims=True)
        var = lib_.var(a.data, axis=norm_axes, keepdims=True)
        rstd = 1.0 / lib_.sqrt(var + self.eps)
        xhat = (a.data - mean) * rstd

        out = xhat
        if self.has_weight:
            out = out * w.data.reshape(_broadcast_shape(a.ndim, self.normalized_shape))
        if self.has_bias:
            out = out + b.data.reshape(_broadcast_shape(a.ndim, self.normalized_shape))

        self._xhat = xhat
        self._rstd = rstd
        self._norm_axes = norm_axes
        self._n = n

        self.result = Tensor(out, device=device)
        return self.result, functools.partial(self.__grad__, a=a, w=w, lib_=lib_)

    def __grad__(self, a: Tensor, w: Tensor, lib_: ModuleType) -> _GradType:
        if self.result is None or self.result.grad is None:
            raise RuntimeError("layer_norm backward called before forward.")

        if self._xhat is None or self._rstd is None or self._norm_axes is None:
            raise RuntimeError("layer_norm cached data missing.")

        dy = self.result.grad
        xhat = self._xhat
        rstd = self._rstd
        norm_axes = self._norm_axes
        n = self._n if self._n is not None else int(np.prod(self.normalized_shape))

        if self.has_weight:
            w_broadcast = w.data.reshape(
                _broadcast_shape(a.ndim, self.normalized_shape)
            )
            dyw = dy * w_broadcast
        else:
            dyw = dy

        sum1 = lib_.sum(dyw, axis=norm_axes, keepdims=True)
        sum2 = lib_.sum(dyw * xhat, axis=norm_axes, keepdims=True)

        dx = (1.0 / n) * rstd * (n * dyw - sum1 - xhat * sum2)

        reduce_axes = tuple(range(0, a.ndim - len(self.normalized_shape)))
        if reduce_axes:
            dweight = lib_.sum(dy * xhat, axis=reduce_axes)
            dbias = lib_.sum(dy, axis=reduce_axes)
        else:
            dweight = dy * xhat
            dbias = dy

        return dx, dweight, dbias


class batch_norm_kernel(Operation):
    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        training: bool = True,
        has_running: bool = True,
        has_weight: bool = True,
        has_bias: bool = True,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.training = bool(training)
        self.has_running = bool(has_running)
        self.has_weight = bool(has_weight)
        self.has_bias = bool(has_bias)

        self._xhat = None
        self._rstd = None
        self._axes = None
        self._m = None
        self._use_batch_stats = None

    def clear(self) -> None:
        super().clear()
        self._xhat = None
        self._rstd = None
        self._axes = None
        self._m = None
        self._use_batch_stats = None

    @func_op(n_in=5, n_ret=1, device="cpu")
    def cpu(
        self, a: Tensor, running_mean: Tensor, running_var: Tensor, w: Tensor, b: Tensor
    ) -> _FuncOpReturnType:
        return self._forward(a, running_mean, running_var, w, b, lib_=np, device="cpu")

    @func_op(n_in=5, n_ret=1, device="gpu")
    def gpu(
        self, a: Tensor, running_mean: Tensor, running_var: Tensor, w: Tensor, b: Tensor
    ) -> _FuncOpReturnType:
        return self._forward(a, running_mean, running_var, w, b, lib_=mx, device="gpu")

    def _forward(
        self,
        a: Tensor,
        running_mean: Tensor,
        running_var: Tensor,
        w: Tensor,
        b: Tensor,
        lib_: ModuleType,
        device: _DeviceType,
    ) -> _FuncOpReturnType:
        axes = (0,) + tuple(range(2, a.ndim))
        m = int(np.prod([a.shape[i] for i in axes]))
        use_batch_stats = self.training or not self.has_running

        if use_batch_stats:
            mean = lib_.mean(a.data, axis=axes, keepdims=True)
            var = lib_.var(a.data, axis=axes, keepdims=True)

            if self.training and self.has_running:
                rm = (
                    self.momentum * mean.reshape(-1)
                    + (1 - self.momentum) * running_mean.data
                )
                rv = (
                    self.momentum * var.reshape(-1)
                    + (1 - self.momentum) * running_var.data
                )
                running_mean.data = rm
                running_var.data = rv

        else:
            mean = running_mean.data.reshape(1, -1, *([1] * (a.ndim - 2)))
            var = running_var.data.reshape(1, -1, *([1] * (a.ndim - 2)))

        rstd = 1.0 / lib_.sqrt(var + self.eps)
        xhat = (a.data - mean) * rstd

        out = xhat
        if self.has_weight:
            out = out * w.data.reshape(1, -1, *([1] * (a.ndim - 2)))
        if self.has_bias:
            out = out + b.data.reshape(1, -1, *([1] * (a.ndim - 2)))

        self._xhat = xhat
        self._rstd = rstd
        self._axes = axes
        self._m = m
        self._use_batch_stats = use_batch_stats

        self.result = Tensor(out, device=device)
        return self.result, functools.partial(self.__grad__, a=a, w=w, lib_=lib_)

    def __grad__(self, a: Tensor, w: Tensor, lib_: ModuleType) -> _GradType:
        if self.result is None or self.result.grad is None:
            raise RuntimeError("batch_norm backward called before forward.")

        if self._rstd is None or self._axes is None or self._m is None:
            raise RuntimeError("batch_norm cached data missing.")

        dy = self.result.grad
        axes = self._axes
        m = self._m

        if self.has_weight:
            w_broadcast = w.data.reshape(1, -1, *([1] * (a.ndim - 2)))
            dyw = dy * w_broadcast
        else:
            dyw = dy

        if self._use_batch_stats:
            xhat = self._xhat
            rstd = self._rstd
            sum1 = lib_.sum(dyw, axis=axes, keepdims=True)
            sum2 = lib_.sum(dyw * xhat, axis=axes, keepdims=True)
            dx = (1.0 / m) * rstd * (m * dyw - sum1 - xhat * sum2)
        else:
            rstd = self._rstd
            dx = dyw * rstd

        reduce_axes = (0,) + tuple(range(2, a.ndim))
        dweight = lib_.sum(
            dy * (self._xhat if self._xhat is not None else 1.0), axis=reduce_axes
        )
        dbias = lib_.sum(dy, axis=reduce_axes)

        return dx, None, None, dweight, dbias


class group_norm_kernel(Operation):
    def __init__(
        self,
        num_groups: int,
        eps: float = 1e-5,
        has_weight: bool = True,
        has_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_groups = int(num_groups)
        self.eps = float(eps)
        self.has_weight = bool(has_weight)
        self.has_bias = bool(has_bias)

        self._xhat = None
        self._rstd = None
        self._group_shape = None
        self._reduce_axes = None
        self._m = None

    def clear(self) -> None:
        super().clear()
        self._xhat = None
        self._rstd = None
        self._group_shape = None
        self._reduce_axes = None
        self._m = None

    @func_op(n_in=3, n_ret=1, device="cpu")
    def cpu(self, a: Tensor, w: Tensor, b: Tensor) -> _FuncOpReturnType:
        return self._forward(a, w, b, lib_=np, device="cpu")

    @func_op(n_in=3, n_ret=1, device="gpu")
    def gpu(self, a: Tensor, w: Tensor, b: Tensor) -> _FuncOpReturnType:
        return self._forward(a, w, b, lib_=mx, device="gpu")

    def _forward(
        self,
        a: Tensor,
        w: Tensor,
        b: Tensor,
        lib_: ModuleType,
        device: _DeviceType,
    ) -> _FuncOpReturnType:
        N, C, *spatial = a.shape
        if C % self.num_groups != 0:
            raise ValueError("num_groups must divide channels.")

        group_size = C // self.num_groups
        group_shape = (N, self.num_groups, group_size, *spatial)
        x = a.data.reshape(group_shape)
        reduce_axes = (2,) + tuple(range(3, x.ndim))
        m = int(np.prod([x.shape[i] for i in reduce_axes]))

        mean = lib_.mean(x, axis=reduce_axes, keepdims=True)
        var = lib_.var(x, axis=reduce_axes, keepdims=True)
        rstd = 1.0 / lib_.sqrt(var + self.eps)
        xhat = (x - mean) * rstd

        out = xhat.reshape(a.shape)
        if self.has_weight:
            out = out * w.data.reshape(1, C, *([1] * len(spatial)))
        if self.has_bias:
            out = out + b.data.reshape(1, C, *([1] * len(spatial)))

        self._xhat = xhat
        self._rstd = rstd
        self._group_shape = group_shape
        self._reduce_axes = reduce_axes
        self._m = m

        self.result = Tensor(out, device=device)
        return self.result, functools.partial(self.__grad__, a=a, w=w, b=b, lib_=lib_)

    def __grad__(self, a: Tensor, w: Tensor, b: Tensor, lib_: ModuleType) -> _GradType:
        if self.result is None or self.result.grad is None:
            raise RuntimeError("group_norm backward called before forward.")

        if (
            self._xhat is None
            or self._rstd is None
            or self._group_shape is None
            or self._reduce_axes is None
            or self._m is None
        ):
            raise RuntimeError("group_norm cached data missing.")

        dy = self.result.grad
        N, C, *spatial = a.shape
        dy_g = dy.reshape(self._group_shape)
        xhat = self._xhat
        rstd = self._rstd
        axes = self._reduce_axes
        m = self._m

        if self.has_weight:
            w_broadcast = w.data.reshape(1, C, *([1] * len(spatial)))
            dyw = dy * w_broadcast
            dyw_g = dyw.reshape(self._group_shape)
        else:
            dyw_g = dy_g

        sum1 = lib_.sum(dyw_g, axis=axes, keepdims=True)
        sum2 = lib_.sum(dyw_g * xhat, axis=axes, keepdims=True)
        dx_g = (1.0 / m) * rstd * (m * dyw_g - sum1 - xhat * sum2)
        dx = dx_g.reshape(a.shape)

        reduce_axes = (0,) + tuple(range(2, a.ndim))
        dweight = lib_.sum(dy * xhat.reshape(a.shape), axis=reduce_axes)
        dbias = lib_.sum(dy, axis=reduce_axes)

        return dx, dweight, dbias
