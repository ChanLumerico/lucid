from typing import Iterable
import numpy as np

import lucid
import lucid.nn as nn
import lucid.optim as optim

from lucid.types import _OptimClosure, _Scalar
from lucid._tensor import Tensor
from lucid._backend.metal import mx


__all__ = ["RMSprop", "Rprop"]


class RMSprop(optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )
        super().__init__(params, defaults)

    def step(self, closure: _OptimClosure | None = None) -> None:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group.get("lr", self.defaults["lr"])
            alpha = group.get("alpha", self.defaults["alpha"])
            eps = group.get("eps", self.defaults["eps"])
            weight_decay = group.get("weight_decay", self.defaults["weight_decay"])
            momentum = group.get("momentum", self.defaults["momentum"])
            centered = group.get("centered", self.defaults["centered"])

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = Tensor.copy_grad(param.grad)
                if weight_decay != 0.0:
                    grad += weight_decay * param.data

                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0
                    state["square_avg"] = lucid.zeros_like(param).data
                    if momentum != 0.0:
                        state["momentum_buffer"] = lucid.zeros_like(param).data
                    if centered:
                        state["grad_avg"] = lucid.zeros_like(param).data

                state["step"] += 1

                square_avg = state["square_avg"]
                square_avg[:] = alpha * square_avg + (1 - alpha) * (grad**2)

                if centered:
                    grad_avg = state["grad_avg"]
                    grad_avg[:] = alpha * grad_avg + (1 - alpha) * grad
                    avg = square_avg - grad_avg**2
                else:
                    avg = square_avg

                denom = lucid.sqrt(avg + eps).data
                if momentum != 0.0:
                    buf = state["momentum_buffer"]
                    buf[:] = momentum * buf + grad / denom
                    update = buf
                else:
                    update = grad / denom

                param.data -= lr * update

        return loss


class Rprop(optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-2,
        etas: tuple[_Scalar, _Scalar] = (0.5, 1.2),
        step_sizes: tuple[_Scalar, _Scalar] = (1e-6, 50.0),
    ) -> None:
        defaults = dict(lr=lr, etas=etas, step_sizes=step_sizes)
        super().__init__(params, defaults)

    def step(self, closure: _OptimClosure | None = None) -> None:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group.get("lr", self.defaults["lr"])
            etaminus, etaplus = group.get("etas", self.defaults["etas"])
            step_min, step_max = group.get("step_sizes", self.defaults["step_sizes"])

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = Tensor.copy_grad(param.grad)
                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0
                    state["prev_grad"] = lucid.zeros_like(param).data
                    state["step_size"] = lucid.ones_like(param).data * lr

                state["step"] += 1

                step_size = state["step_size"]
                prev_grad = state["prev_grad"]

                sign_change = grad * prev_grad

                lib_ = np if param.is_cpu() else mx
                step_size = lib_.where(sign_change > 0, step_size * etaplus, step_size)
                step_size = lib_.where(sign_change < 0, step_size * etaminus, step_size)

                step_size[:] = lucid.clip(step_size, step_min, step_max).data

                grad = lib_.where(sign_change < 0, 0, grad)
                state["prev_grad"] = Tensor.copy_grad(grad)

                param.data -= lucid.sign(grad).data * step_size

        return loss
