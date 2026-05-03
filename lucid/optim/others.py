"""
Additional optimizers: RMSprop, Adagrad, Adadelta, Adamax, RAdam, NAdam, ASGD, Rprop.
"""

from typing import Any
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap
from lucid.optim.optimizer import Optimizer


def _make_engine_optim(engine_cls: type, keys: list[str], defaults: dict[str, Any]) -> type:
    """Factory to create a simple engine-backed Optimizer subclass."""

    class _EngineOptim(Optimizer):
        def __init__(self, params: Any, **kwargs: Any) -> None:
            super().__init__(params, {**defaults, **kwargs})

        def _rebuild_engine_optims(self) -> None:
            self._engine_optims = [
                engine_cls(*([_unwrap(p) for p in g["params"]] + [g[k] for k in keys]))
                for g in self.param_groups
            ]

        def step(self, closure: Any = None) -> Any:
            loss = closure() if closure is not None else None
            for optim in self._engine_optims:
                optim.step()
            return loss

    return _EngineOptim


class RMSprop(Optimizer):
    """RMSprop optimizer."""

    def __init__(self, params: Any, lr: float = 1e-2, alpha: float = 0.99,
                 eps: float = 1e-8, weight_decay: float = 0, momentum: float = 0,
                 centered: bool = False) -> None:
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,
                        momentum=momentum, centered=centered)
        super().__init__(params, defaults)

    def _rebuild_engine_optims(self) -> None:
        self._engine_optims = [
            _C_engine.RMSprop(
                [_unwrap(p) for p in g["params"]],
                g["lr"], g.get("alpha", 0.99), g.get("eps", 1e-8),
                g.get("weight_decay", 0), g.get("momentum", 0),
            )
            for g in self.param_groups
        ]

    def step(self, closure: Any = None) -> Any:
        loss = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()
        return loss


class Adagrad(Optimizer):
    """Adagrad optimizer."""

    def __init__(self, params: Any, lr: float = 1e-2, lr_decay: float = 0,
                 weight_decay: float = 0, eps: float = 1e-10) -> None:
        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    def _rebuild_engine_optims(self) -> None:
        self._engine_optims = [
            _C_engine.Adagrad(
                [_unwrap(p) for p in g["params"]],
                g["lr"], g.get("lr_decay", 0), g.get("weight_decay", 0), g.get("eps", 1e-10),
            )
            for g in self.param_groups
        ]

    def step(self, closure: Any = None) -> Any:
        loss = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()
        return loss


class Adadelta(Optimizer):
    """Adadelta optimizer."""

    def __init__(self, params: Any, lr: float = 1.0, rho: float = 0.9,
                 eps: float = 1e-6, weight_decay: float = 0) -> None:
        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _rebuild_engine_optims(self) -> None:
        self._engine_optims = [
            _C_engine.Adadelta(
                [_unwrap(p) for p in g["params"]],
                g["lr"], g.get("rho", 0.9), g.get("eps", 1e-6), g.get("weight_decay", 0),
            )
            for g in self.param_groups
        ]

    def step(self, closure: Any = None) -> Any:
        loss = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()
        return loss


class Adamax(Optimizer):
    """Adamax optimizer (variant of Adam based on infinity norm)."""

    def __init__(self, params: Any, lr: float = 2e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0) -> None:
        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _rebuild_engine_optims(self) -> None:
        self._engine_optims = [
            _C_engine.Adamax(
                [_unwrap(p) for p in g["params"]],
                g["lr"], g.get("beta1", 0.9), g.get("beta2", 0.999),
                g.get("eps", 1e-8), g.get("weight_decay", 0),
            )
            for g in self.param_groups
        ]

    def step(self, closure: Any = None) -> Any:
        loss = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()
        return loss


class RAdam(Optimizer):
    """Rectified Adam optimizer."""

    def __init__(self, params: Any, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0) -> None:
        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _rebuild_engine_optims(self) -> None:
        self._engine_optims = [
            _C_engine.RAdam(
                [_unwrap(p) for p in g["params"]],
                g["lr"], g.get("beta1", 0.9), g.get("beta2", 0.999),
                g.get("eps", 1e-8), g.get("weight_decay", 0),
            )
            for g in self.param_groups
        ]

    def step(self, closure: Any = None) -> Any:
        loss = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()
        return loss


class NAdam(Optimizer):
    """Nesterov Adam optimizer."""

    def __init__(self, params: Any, lr: float = 2e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0) -> None:
        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _rebuild_engine_optims(self) -> None:
        self._engine_optims = [
            _C_engine.NAdam(
                [_unwrap(p) for p in g["params"]],
                g["lr"], g.get("beta1", 0.9), g.get("beta2", 0.999),
                g.get("eps", 1e-8), g.get("weight_decay", 0),
            )
            for g in self.param_groups
        ]

    def step(self, closure: Any = None) -> Any:
        loss = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()
        return loss


class ASGD(Optimizer):
    """Averaged stochastic gradient descent."""

    def __init__(self, params: Any, lr: float = 1e-2, lambd: float = 1e-4,
                 alpha: float = 0.75, t0: float = 1e6, weight_decay: float = 0) -> None:
        defaults = dict(lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _rebuild_engine_optims(self) -> None:
        self._engine_optims = [
            _C_engine.ASGD(
                [_unwrap(p) for p in g["params"]],
                g["lr"], g.get("lambd", 1e-4), g.get("alpha", 0.75),
                g.get("t0", 1e6), g.get("weight_decay", 0),
            )
            for g in self.param_groups
        ]

    def step(self, closure: Any = None) -> Any:
        loss = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()
        return loss


class Rprop(Optimizer):
    """Resilient backpropagation optimizer."""

    def __init__(self, params: Any, lr: float = 1e-2, etas: tuple[float, float] = (0.5, 1.2),
                 step_sizes: tuple[float, float] = (1e-6, 50)) -> None:
        defaults = dict(lr=lr, eta_minus=etas[0], eta_plus=etas[1],
                        step_min=step_sizes[0], step_max=step_sizes[1])
        super().__init__(params, defaults)

    def _rebuild_engine_optims(self) -> None:
        self._engine_optims = [
            _C_engine.Rprop(
                [_unwrap(p) for p in g["params"]],
                g["lr"], g.get("eta_minus", 0.5), g.get("eta_plus", 1.2),
                g.get("step_min", 1e-6), g.get("step_max", 50),
            )
            for g in self.param_groups
        ]

    def step(self, closure: Any = None) -> Any:
        loss = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()
        return loss
