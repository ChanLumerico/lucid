from typing import Iterable

import lucid
import lucid.nn as nn
import lucid.optim as optim

from lucid.types import _OptimClosure, _Scalar


__all__ = ["Adam"]


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

                grad = param.grad.copy()
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
