from typing import Iterable

import lucid
import lucid.nn as nn
import lucid.optim as optim

from lucid.types import _OptimClosure


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

    def step(self, closure: _OptimClosure | None = None) -> None:  # Beta
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

                grad = param.grad.copy()
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


class Rprop(optim.Optimizer): ...
