import functools
from types import ModuleType

import numpy as np

from lucid._backend.core import Operation, func_op, _FuncOpReturnType, _GradType
from lucid._backend.metal import mx
from lucid._tensor import Tensor
from lucid.types import _DeviceType, _TensorData


def _to_int(arr: _TensorData, lib_: ModuleType) -> _TensorData:
    if lib_ is np:
        return arr.astype(np.int64)
    return arr.astype(mx.int32)


class cross_entropy_kernel(Operation):
    def __init__(
        self,
        reduction: str | None = "mean",
        eps: float = 1e-7,
        ignore_index: int | None = None,
        has_weight: bool = True,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.eps = float(eps)
        self.ignore_index = ignore_index
        self.has_weight = bool(has_weight)

        self._log_probs = None
        self._probs = None
        self._target = None
        self._weight = None
        self._valid_count = None

    def clear(self) -> None:
        super().clear()
        self._log_probs = None
        self._probs = None
        self._target = None
        self._weight = None
        self._valid_count = None

    @func_op(n_in=3, n_ret=1)
    def cpu(self, logits: Tensor, target: Tensor, weight: Tensor) -> _FuncOpReturnType:
        return self._forward(logits, target, weight, lib_=np, device="cpu")

    @func_op(n_in=3, n_ret=1, device="gpu")
    def gpu(self, logits: Tensor, target: Tensor, weight: Tensor) -> _FuncOpReturnType:
        return self._forward(logits, target, weight, lib_=mx, device="gpu")

    def _forward(
        self,
        logits: Tensor,
        target: Tensor,
        weight: Tensor,
        lib_: ModuleType,
        device: _DeviceType,
    ) -> _FuncOpReturnType:
        if logits.ndim != 2:
            raise ValueError("cross_entropy expects 2D logits [N, C].")

        N, _ = logits.shape
        max_val = lib_.max(logits.data, axis=1, keepdims=True)
        exp_x = lib_.exp(logits.data - max_val)
        sum_exp = lib_.sum(exp_x, axis=1, keepdims=True)
        log_probs = (logits.data - max_val) - lib_.log(sum_exp)
        probs = exp_x / sum_exp

        target_int = _to_int(target.data, lib_)
        if lib_ is np:
            idx = np.arange(N, dtype=np.int64)
        else:
            idx = mx.arange(N, dtype=mx.int32)

        if self.ignore_index is not None:
            mask = target_int != self.ignore_index
        else:
            mask = None

        gather = log_probs[idx, target_int]
        loss = -gather

        if self.has_weight:
            w = weight.data
            loss = loss * w[target_int]
        else:
            w = None

        if mask is not None:
            if lib_ is np:
                loss = loss * mask.astype(loss.dtype)
            else:
                loss = loss * mask.astype(loss.dtype)

        if self.reduction is None:
            out = loss
            valid_count = None
        elif self.reduction == "sum":
            out = lib_.sum(loss)
            valid_count = None
        elif self.reduction == "mean":
            if mask is None:
                valid_count = N
            else:
                valid_count = lib_.sum(mask)
                if hasattr(valid_count, "item") and valid_count.item() == 0:
                    out = lib_.zeros((), dtype=loss.dtype)
                    self.result = Tensor(out, device=device)
                    return self.result, functools.partial(self.__grad__, lib_=lib_)
            out = lib_.sum(loss) / valid_count
        else:
            raise ValueError("Invalid reduction type. Choose 'mean', 'sum', or 'none'.")

        self._log_probs = log_probs
        self._probs = probs
        self._target = target_int
        self._weight = w
        self._valid_count = valid_count

        self.result = Tensor(out, device=device)
        return self.result, functools.partial(self.__grad__, lib_=lib_)

    def __grad__(self, lib_: ModuleType) -> _GradType:
        if self.result is None or self.result.grad is None:
            raise RuntimeError("cross_entropy backward called before forward.")
        if self._probs is None or self._target is None:
            raise RuntimeError("cross_entropy cached data missing.")

        probs = self._probs
        target = self._target
        N = probs.shape[0]
        C = probs.shape[1]

        if lib_ is np:
            grad_input = probs.copy()
            idx = np.arange(N, dtype=np.int64)
        else:
            grad_input = mx.array(probs)
            idx = mx.arange(N, dtype=mx.int32)

        grad_input[idx, target] = grad_input[idx, target] - 1

        if self._weight is not None:
            grad_input = grad_input * self._weight[target][:, None]

        if self.ignore_index is not None:
            if lib_ is np:
                mask = (target != self.ignore_index).astype(grad_input.dtype)
                grad_input = grad_input * mask[:, None]
            else:
                mask = (target != self.ignore_index).astype(grad_input.dtype)
                grad_input = grad_input * mask[:, None]

        if self.reduction is None:
            go = self.result.grad
            grad_input = grad_input * go[:, None]
            dweight = None
        else:
            go = self.result.grad
            if self.reduction == "mean":
                if self._valid_count is None:
                    grad_input = grad_input / N
                else:
                    grad_input = grad_input / self._valid_count
            grad_input = grad_input * go

            dweight = None
            if self._weight is not None:
                if lib_ is np:
                    dweight = np.zeros((C,), dtype=grad_input.dtype)
                else:
                    dweight = mx.zeros((C,), dtype=grad_input.dtype)

                if self.reduction is None:
                    go_vec = go
                else:
                    go_vec = None

                for c in range(C):
                    if lib_ is np:
                        mask_c = target == c
                        if self.ignore_index is not None:
                            mask_c = mask_c & (target != self.ignore_index)
                        if go_vec is None:
                            contrib = -self._log_probs[mask_c, c]
                            if self.reduction == "mean":
                                denom = (
                                    self._valid_count
                                    if self._valid_count is not None
                                    else N
                                )
                                contrib = contrib / denom
                            dweight[c] = np.sum(contrib) * go
                        else:
                            contrib = -self._log_probs[mask_c, c]
                            dweight[c] = np.sum(contrib * go_vec[mask_c])
                    else:
                        mask_c = target == c
                        if self.ignore_index is not None:
                            mask_c = mask_c & (target != self.ignore_index)

                        contrib = -self._log_probs[mask_c, c]
                        if self.reduction == "mean":
                            denom = (
                                self._valid_count
                                if self._valid_count is not None
                                else N
                            )
                            contrib = contrib / denom

                        if go_vec is None:
                            dweight = dweight.at[c].add(mx.sum(contrib) * go)
                        else:
                            dweight = dweight.at[c].add(
                                mx.sum(contrib * go_vec[mask_c])
                            )

        return grad_input, None, dweight


class binary_cross_entropy_kernel(Operation):
    def __init__(
        self,
        reduction: str | None = "mean",
        eps: float = 1e-7,
        has_weight: bool = True,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.eps = float(eps)
        self.has_weight = bool(has_weight)

        self._input = None
        self._target = None
        self._weight = None

    def clear(self) -> None:
        super().clear()
        self._input = None
        self._target = None
        self._weight = None

    @func_op(n_in=3, n_ret=1)
    def cpu(self, input_: Tensor, target: Tensor, weight: Tensor) -> _FuncOpReturnType:
        return self._forward(input_, target, weight, lib_=np, device="cpu")

    @func_op(n_in=3, n_ret=1, device="gpu")
    def gpu(self, input_: Tensor, target: Tensor, weight: Tensor) -> _FuncOpReturnType:
        return self._forward(input_, target, weight, lib_=mx, device="gpu")

    def _forward(
        self,
        input_: Tensor,
        target: Tensor,
        weight: Tensor,
        lib_: ModuleType,
        device: _DeviceType,
    ) -> _FuncOpReturnType:
        x = input_.data
        t = target.data
        x = lib_.clip(x, self.eps, 1.0 - self.eps)
        loss = -(t * lib_.log(x) + (1 - t) * lib_.log(1 - x))

        if self.has_weight:
            loss = loss * weight.data

        if self.reduction is None:
            out = loss
        elif self.reduction == "sum":
            out = lib_.sum(loss)
        elif self.reduction == "mean":
            out = lib_.mean(loss)
        else:
            raise ValueError("Invalid reduction type. Choose 'mean', 'sum', or 'none'.")

        self._input = x
        self._target = t
        self._weight = weight.data if self.has_weight else None

        self.result = Tensor(out, device=device)
        return self.result, functools.partial(self.__grad__)

    def __grad__(self) -> _GradType:
        if self.result is None or self.result.grad is None:
            raise RuntimeError("binary_cross_entropy backward called before forward.")
        if self._input is None or self._target is None:
            raise RuntimeError("binary_cross_entropy cached data missing.")

        x = self._input
        t = self._target
        grad = (x - t) / (x * (1 - x))

        if self._weight is not None:
            grad = grad * self._weight

        if self.reduction is None:
            grad = grad * self.result.grad
        elif self.reduction == "sum":
            grad = grad * self.result.grad
        elif self.reduction == "mean":
            grad = grad * (self.result.grad / x.size)

        return grad, None, None


class binary_cross_entropy_with_logits_kernel(Operation):
    def __init__(
        self,
        reduction: str | None = "mean",
        has_weight: bool = True,
        has_pos_weight: bool = True,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.has_weight = bool(has_weight)
        self.has_pos_weight = bool(has_pos_weight)

        self._logits = None
        self._target = None
        self._weight = None
        self._pos_weight = None

    def clear(self) -> None:
        super().clear()
        self._logits = None
        self._target = None
        self._weight = None
        self._pos_weight = None

    @func_op(n_in=4, n_ret=1)
    def cpu(
        self, logits: Tensor, target: Tensor, weight: Tensor, pos_weight: Tensor
    ) -> _FuncOpReturnType:
        return self._forward(logits, target, weight, pos_weight, lib_=np, device="cpu")

    @func_op(n_in=4, n_ret=1, device="gpu")
    def gpu(
        self, logits: Tensor, target: Tensor, weight: Tensor, pos_weight: Tensor
    ) -> _FuncOpReturnType:
        return self._forward(logits, target, weight, pos_weight, lib_=mx, device="gpu")

    def _forward(
        self,
        logits: Tensor,
        target: Tensor,
        weight: Tensor,
        pos_weight: Tensor,
        lib_: ModuleType,
        device: _DeviceType,
    ) -> _FuncOpReturnType:
        x = logits.data
        t = target.data

        max_val = lib_.maximum(-x, 0)
        sp = max_val + lib_.log(lib_.exp(-max_val) + lib_.exp(-x - max_val))

        if self.has_pos_weight:
            pw = pos_weight.data
            coeff = 1 + (pw - 1) * t
            loss = (1 - t) * x + coeff * sp
        else:
            pw = None
            loss = lib_.maximum(x, 0) - x * t + sp

        if self.has_weight:
            loss = loss * weight.data

        if self.reduction is None:
            out = loss
        elif self.reduction == "sum":
            out = lib_.sum(loss)
        elif self.reduction == "mean":
            out = lib_.mean(loss)
        else:
            raise ValueError("Invalid reduction type. Choose 'mean', 'sum', or 'none'.")

        self._logits = x
        self._target = t
        self._weight = weight.data if self.has_weight else None
        self._pos_weight = pw

        self.result = Tensor(out, device=device)
        return self.result, functools.partial(self.__grad__, lib_=lib_)

    def __grad__(self, lib_: ModuleType) -> _GradType:
        if self.result is None or self.result.grad is None:
            raise RuntimeError(
                "binary_cross_entropy_with_logits backward called before forward."
            )
        if self._logits is None or self._target is None:
            raise RuntimeError("binary_cross_entropy_with_logits cached data missing.")

        x = self._logits
        t = self._target
        sig = 1.0 / (1.0 + lib_.exp(-x))

        if self._pos_weight is not None:
            pw = self._pos_weight
            grad = (sig - t) * (1 + (pw - 1) * t)
        else:
            grad = sig - t

        if self._weight is not None:
            grad = grad * self._weight

        if self.reduction is None:
            grad = grad * self.result.grad
        elif self.reduction == "sum":
            grad = grad * self.result.grad
        elif self.reduction == "mean":
            grad = grad * (self.result.grad / x.size)

        return grad, None, None, None
