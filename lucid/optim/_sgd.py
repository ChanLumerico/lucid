import copy
from typing import Any, Iterable

import lucid
import lucid.nn as nn
import lucid.optim as optim

from lucid.types import _OptimClosure


__all__ = ["SGD", "ASGD"]


class SGD(optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: _OptimClosure | None = None) -> Any | None:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group.get("lr", self.defaults["lr"])
            momentum = group.get("momentum", self.defaults["momentum"])
            weight_decay = group.get("weight_decay", self.defaults["weight_decay"])

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.copy()
                if weight_decay != 0:
                    grad = grad + weight_decay * param.data

                if momentum != 0:
                    param_state = self.state[param]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = copy.deepcopy(grad)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf = momentum * buf + grad
                        param_state["momentum_buffer"] = buf

                    grad = buf

                param.data = param.data - lr * grad

        return loss


class ASGD(optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        alpha: float = 0.75,
        t0: float = 1e6,
        lambd: float = 1e-4,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=alpha,
            t0=t0,
            lambd=lambd,
        )
        super().__init__(params, defaults)

    def step(self, closure: _OptimClosure | None = None) -> Any | None:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group.get("lr", self.defaults["lr"])
            momentum = group.get("momentum", self.defaults["momentum"])
            weight_decay = group.get("weight_decay", self.defaults["weight_decay"])
            alpha = group.get("alpha", self.defaults["alpha"])
            t0 = group.get("t0", self.defaults["t0"])
            lambd = group.get("lambd", self.defaults["lambd"])

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.copy()
                if weight_decay != 0:
                    grad = grad + weight_decay * param.data
                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0
                    if momentum != 0:
                        state["momentum_buffer"] = copy.deepcopy(grad)
                    state["ax"] = copy.deepcopy(param.data)

                state["step"] += 1
                if momentum != 0:
                    buf = state["momentum_buffer"]
                    buf = momentum * buf + grad
                    state["momentum_buffer"] = buf
                    grad = buf

                param.data = param.data - lr * grad

                step = state["step"]
                if step >= t0:
                    ax = state["ax"]
                    coef = 1.0 / (alpha * step + 1)
                    ax = (1.0 - coef) * ax + coef * param.data - lambd * ax
                    state["ax"] = ax

        return loss

    def get_averages(self) -> Iterable[nn.Parameter]:
        averages = []
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                if "ax" in state:
                    avg_param = state["ax"]
                    averages.append(avg_param)
                else:
                    averages.append(param.data)

        return averages
