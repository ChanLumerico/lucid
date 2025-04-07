from typing import Iterable

import lucid
import lucid.nn as nn
import lucid.optim as optim

from lucid.types import _OptimClosure, _Scalar
from lucid._tensor import Tensor


__all__ = ["Adam", "AdamW", "NAdam", "RAdam"]


class Adam(optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[_Scalar, _Scalar] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

    def step(self, closure: _OptimClosure | None = None) -> None:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group.get("lr", self.defaults["lr"])
            beta1, beta2 = group.get("betas", self.defaults["betas"])
            eps = group.get("eps", self.defaults["eps"])
            weight_decay = group.get("weight_decay", self.defaults["weight_decay"])
            amsgrad = group.get("amsgrad", self.defaults["amsgrad"])

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = Tensor.copy_grad(param.grad)
                if weight_decay != 0.0:
                    grad += weight_decay * param.data

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = lucid.zeros_like(param).data
                    state["exp_avg_sq"] = lucid.zeros_like(param).data
                    if amsgrad:
                        state["max_exp_avg_sq"] = lucid.zeros_like(param).data

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]

                exp_avg[:] = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq[:] = beta2 * exp_avg_sq + (1 - beta2) * (grad**2)

                if amsgrad:
                    max_exp_avg_sq = lucid.maximum(max_exp_avg_sq, exp_avg_sq)
                    denom = lucid.sqrt(max_exp_avg_sq) + eps
                else:
                    denom = lucid.sqrt(exp_avg_sq) + eps

                bias_correct1 = 1 - beta1 ** state["step"]
                bias_correct2 = 1 - beta2 ** state["step"]

                step_size = lr * (bias_correct2**0.5) / bias_correct1
                param.data -= step_size * (exp_avg / denom.data)

        return loss


class AdamW(optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[_Scalar, _Scalar] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

    def step(self, closure: _OptimClosure | None = None) -> None:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group.get("lr", self.defaults["lr"])
            beta1, beta2 = group.get("betas", self.defaults["betas"])
            eps = group.get("eps", self.defaults["eps"])
            weight_decay = group.get("weight_decay", self.defaults["weight_decay"])
            amsgrad = group.get("amsgrad", self.defaults["amsgrad"])

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = Tensor.copy_grad(param.grad)
                if weight_decay != 0.0:
                    param.data -= lr * weight_decay * param.data

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = lucid.zeros_like(param).data
                    state["exp_avg_sq"] = lucid.zeros_like(param).data
                    if amsgrad:
                        state["max_exp_avg_sq"] = lucid.zeros_like(param).data

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]

                exp_avg[:] = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq[:] = beta2 * exp_avg_sq + (1 - beta2) * (grad**2)

                if amsgrad:
                    max_exp_avg_sq = lucid.maximum(max_exp_avg_sq, exp_avg_sq)
                    denom = lucid.sqrt(max_exp_avg_sq) + eps
                else:
                    denom = lucid.sqrt(exp_avg_sq) + eps

                bias_correct1 = 1 - beta1 ** state["step"]
                bias_correct2 = 1 - beta2 ** state["step"]

                step_size = lr * (bias_correct2**0.5) / bias_correct1
                param.data -= step_size * (exp_avg / denom.data)

        return loss


class NAdam(optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 2e-3,
        betas: tuple[_Scalar, _Scalar] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum_decay: float = 0.004,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum_decay=momentum_decay,
        )
        super().__init__(params, defaults)

    def step(self, closure: _OptimClosure | None = None) -> None:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group.get("lr", self.defaults["lr"])
            beta1, beta2 = group.get("betas", self.defaults["betas"])
            eps = group.get("eps", self.defaults["eps"])
            weight_decay = group.get("weight_decay", self.defaults["weight_decay"])
            momentum_decay = group.get(
                "momentum_decay", self.defaults["momentum_decay"]
            )

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = Tensor.copy_grad(param.grad)
                if weight_decay != 0.0:
                    grad += weight_decay * param.data

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = lucid.zeros_like(param).data
                    state["exp_avg_sq"] = lucid.zeros_like(param).data

                state["step"] += 1

                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg[:] *= 1 - momentum_decay
                exp_avg[:] = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq[:] = beta2 * exp_avg_sq + (1 - beta2) * (grad**2)

                bias_correct1 = 1 - beta1**step
                bias_correct2 = 1 - beta2**step

                m_t_hat = exp_avg / bias_correct1
                v_t_hat = exp_avg_sq / bias_correct2

                lookahead_term = (1 - beta1) / (1 - beta1**step) * grad
                step_size = lr * (bias_correct2**0.5) / bias_correct1
                param.data -= (
                    step_size
                    * (m_t_hat * beta1 + lookahead_term)
                    / (lucid.sqrt(v_t_hat) + eps).data
                )

        return loss


class RAdam(optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[_Scalar, _Scalar] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def step(self, closure: _OptimClosure | None = None) -> None:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group.get("lr", self.defaults["lr"])
            beta1, beta2 = group.get("betas", self.defaults["betas"])
            eps = group.get("eps", self.defaults["eps"])
            weight_decay = group.get("weight_decay", self.defaults["weight_decay"])

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = Tensor.copy_grad(param.grad)
                if weight_decay != 0.0:
                    grad += weight_decay * param.data

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = lucid.zeros_like(param).data
                    state["exp_avg_sq"] = lucid.zeros_like(param).data

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg[:] = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq[:] = beta2 * exp_avg_sq + (1 - beta2) * (grad**2)

                bias_correct1 = 1 - beta1**step
                bias_correct2 = 1 - beta2**step

                m_t_hat = exp_avg / bias_correct1

                rho_inf = 2 / (1 - beta2) - 1
                rho_t = rho_inf - 2 * step * beta2**step / (1 - beta2**step)

                if rho_t > 4:
                    r_t = (
                        (rho_t - 4)
                        * (rho_t - 2)
                        * rho_inf
                        / ((rho_inf - 4) * (rho_inf - 2))
                    ) ** 0.5
                    v_t_hat = exp_avg_sq / bias_correct2
                    step_size = lr * r_t / (lucid.sqrt(v_t_hat) + eps).data
                else:
                    step_size = lr

                param.data -= step_size * m_t_hat

        return loss
