from typing import Iterable

import lucid
import lucid.nn as nn
import lucid.optim as optim

from lucid.types import _OptimClosure, _Scalar


__all__ = ["Adamax", "Adagrad", "Adadelta"]


class Adamax(optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 2e-3,
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

                grad = param.grad.copy()
                if weight_decay != 0.0:
                    grad += weight_decay * param.data

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = lucid.zeros_like(param).data
                    state["exp_inf"] = lucid.zeros_like(param).data

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_inf = state["exp_inf"]

                exp_avg[:] = beta1 * exp_avg + (1 - beta1) * grad
                exp_inf[:] = lucid.maximum(beta2 * exp_inf, lucid.abs(grad)).data

                bias_correct1 = 1 - beta1**step

                step_size = lr / bias_correct1
                param.data -= step_size * (exp_avg / (exp_inf + eps))

        return loss


class Adagrad(optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-2,
        eps: float = 1e-10,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
        )
        super().__init__(params, defaults)

    def step(self, closure: _OptimClosure | None = None) -> None:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group.get("lr", self.defaults["lr"])
            eps = group.get("eps", self.defaults["eps"])
            weight_decay = group.get("weight_decay", self.defaults["weight_decay"])
            initial_accumulator_value = group.get(
                "initial_accumulator_value",
                self.defaults["initial_accumulator_value"],
            )

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.copy()
                if weight_decay != 0.0:
                    grad += weight_decay * param.data

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["sum_sq_grad"] = (
                        lucid.ones_like(param).data * initial_accumulator_value
                    )

                state["step"] += 1
                sum_sq_grad = state["sum_sq_grad"]
                sum_sq_grad[:] += grad**2

                step_size = lr / (lucid.sqrt(sum_sq_grad) + eps).data
                param.data -= step_size * grad

        return loss


class Adadelta(optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: _OptimClosure | None = None) -> None:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group.get("lr", self.defaults["lr"])
            rho = group.get("rho", self.defaults["rho"])
            eps = group.get("eps", self.defaults["eps"])
            weight_decay = group.get("weight_decay", self.defaults["weight_decay"])

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.copy()
                if weight_decay != 0.0:
                    grad += weight_decay * param.data

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["sq_avg"] = lucid.zeros_like(param).data
                    state["accumulated_update"] = lucid.zeros_like(param).data

                state["step"] += 1
                sq_avg = state["sq_avg"]
                accumulated_update = state["accumulated_update"]

                sq_avg[:] = rho * sq_avg + (1 - rho) * (grad**2)
                update = (
                    lucid.sqrt(accumulated_update + eps) / lucid.sqrt(sq_avg + eps)
                ).data * grad

                accumulated_update[:] = rho * accumulated_update + (1 - rho) * (
                    update**2
                )
                param.data -= lr * update

        return loss
