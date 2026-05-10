"""
SGD optimizer.
"""

from typing import Iterable, cast
from lucid._tensor.tensor import Tensor
from lucid._types import _OptimizerClosure
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap
from lucid.nn.parameter import Parameter
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
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
    ) -> None:
        defaults: dict[str, object] = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        from lucid.nn.parameter import Parameter as _P

        params_list: list[_P] = group["params"]  # type: ignore[assignment]  # params is list[Parameter] at runtime
        self._engine_optims.append(
            _C_engine.SGD(
                [_unwrap(p) for p in params_list],
                cast(float, group["lr"]),
                cast(float, group.get("momentum", 0.0)),
                cast(float, group.get("dampening", 0.0)),
                cast(float, group.get("weight_decay", 0.0)),
            )
        )

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single SGD step."""
        loss: Tensor | None = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()  # type: ignore[attr-defined]  # _EngineOptimizer.step() correct at runtime
        return loss
