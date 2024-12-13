import copy
from typing import Any, Callable, Iterable

import lucid
import lucid.nn as nn
import lucid.optim as optim


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

    def step(self, closure: Callable[[], Any] | None = None) -> Any | None:
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

                d_p = param.grad
                if weight_decay != 0:
                    d_p = d_p + weight_decay * param.data

                if momentum != 0:
                    param_state = self.state[param]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = copy.deepcopy(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf = momentum * buf + d_p
                        param_state["momentum_buffer"] = buf

                    d_p = buf

                param.data = param.data - lr * d_p

        return loss


class ASGD(optim.Optimizer): ...
