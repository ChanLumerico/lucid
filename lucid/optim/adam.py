from typing import Iterable

import lucid
import lucid.nn as nn
import lucid.optim as optim

from lucid.types import _OptimClosure, _Scalar
from lucid._tensor import Tensor
from lucid._backend.metal import post_step_eval

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

        with lucid.no_grad():
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
                        exp_avg_sq = max_exp_avg_sq

                    bias_correct1 = 1 - beta1 ** state["step"]
                    bias_correct2 = 1 - beta2 ** state["step"]
                    bias_correct2_sqrt = bias_correct2**0.5

                    step_size = lr * bias_correct2_sqrt / bias_correct1
                    denom = lucid.sqrt(exp_avg_sq) + eps * bias_correct2_sqrt
                    param.data -= step_size * (exp_avg / denom.data)

                    post_step_eval(param, self.state.get(param))

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

        with lucid.no_grad():
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
                        exp_avg_sq = max_exp_avg_sq

                    bias_correct1 = 1 - beta1 ** state["step"]
                    bias_correct2 = 1 - beta2 ** state["step"]
                    bias_correct2_sqrt = bias_correct2**0.5

                    step_size = lr * bias_correct2_sqrt / bias_correct1
                    denom = lucid.sqrt(exp_avg_sq) + eps * bias_correct2_sqrt
                    param.data -= step_size * (exp_avg / denom.data)

                    post_step_eval(param, self.state.get(param))

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

        with lucid.no_grad():
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
                        state["mu_product"] = 1.0

                    state["step"] += 1
                    step = state["step"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    mu = beta1 * (1.0 - 0.5 * (0.96 ** (step * momentum_decay)))
                    mu_next = beta1 * (
                        1.0 - 0.5 * (0.96 ** ((step + 1) * momentum_decay))
                    )
                    state["mu_product"] *= mu
                    mu_product = state["mu_product"]
                    mu_product_next = mu_product * mu_next

                    exp_avg[:] = beta1 * exp_avg + (1 - beta1) * grad
                    exp_avg_sq[:] = beta2 * exp_avg_sq + (1 - beta2) * (grad**2)

                    bias_correct2 = 1 - beta2**step
                    denom = (lucid.sqrt(exp_avg_sq / bias_correct2) + eps).data

                    param.data -= lr * (1.0 - mu) / (1.0 - mu_product) * (grad / denom)
                    param.data -= (
                        lr * mu_next / (1.0 - mu_product_next) * (exp_avg / denom)
                    )

                    post_step_eval(param, self.state.get(param))

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

        with lucid.no_grad():
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
                    bias_correct2_sqrt = bias_correct2**0.5

                    m_t_hat = exp_avg / bias_correct1

                    rho_inf = 2 / (1 - beta2) - 1
                    rho_t = rho_inf - 2 * step * beta2**step / (1 - beta2**step)

                    if rho_t > 5.0:
                        r_t = (
                            (rho_t - 4)
                            * (rho_t - 2)
                            * rho_inf
                            / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                        ) ** 0.5
                        adaptive = (
                            bias_correct2_sqrt / (lucid.sqrt(exp_avg_sq) + eps).data
                        )
                        param.data -= lr * r_t * m_t_hat * adaptive
                    else:
                        param.data -= lr * m_t_hat

                    post_step_eval(param, self.state.get(param))

        return loss
