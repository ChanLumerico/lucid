"""
SGD optimizer.
"""

from typing import Any
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap
from lucid.optim.optimizer import Optimizer


class SGD(Optimizer):
    """
    Stochastic gradient descent (with optional momentum).

    Args:
        params:       iterable of Parameters to optimize
        lr:           learning rate
        momentum:     momentum factor (default: 0)
        dampening:    dampening for momentum (default: 0)
        weight_decay: L2 penalty (default: 0)
        nesterov:     enables Nesterov momentum (default: False)
    """

    def __init__(
        self,
        params: Any,
        lr: float,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr, momentum=momentum, dampening=dampening,
            weight_decay=weight_decay, nesterov=nesterov,
        )
        super().__init__(params, defaults)

    def _rebuild_engine_optims(self) -> None:
        self._engine_optims = [
            _C_engine.SGD(
                [_unwrap(p) for p in g["params"]],
                g["lr"], g.get("momentum", 0), g.get("dampening", 0),
                g.get("weight_decay", 0), g.get("nesterov", False),
            )
            for g in self.param_groups
        ]

    def step(self, closure: Any = None) -> Any:
        loss = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()
        return loss
