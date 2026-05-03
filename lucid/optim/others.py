"""
Additional optimizers: RMSprop, Adagrad, Adadelta, Adamax, RAdam, NAdam, ASGD, Rprop.
"""

from typing import Any
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap
from lucid.optim.optimizer import Optimizer


class RMSprop(Optimizer):
    """RMSprop optimizer."""

    def __init__(self, params: Any, lr: float = 1e-2, alpha: float = 0.99,
                 eps: float = 1e-8, weight_decay: float = 0, momentum: float = 0,
                 centered: bool = False) -> None:
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,
                        momentum=momentum, centered=centered)
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, Any]) -> None:
        self._engine_optims.append(
            _C_engine.RMSprop(
                [_unwrap(p) for p in group["params"]],
                group["lr"],
                group.get("alpha", 0.99),
                group.get("eps", 1e-8),
                group.get("weight_decay", 0.0),
                group.get("momentum", 0.0),
            )
        )

    def step(self, closure: Any = None) -> Any:
        """Perform a single RMSprop step."""
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

    def _append_engine_optim(self, group: dict[str, Any]) -> None:
        self._engine_optims.append(
            _C_engine.Adagrad(
                [_unwrap(p) for p in group["params"]],
                group["lr"],
                group.get("lr_decay", 0.0),
                group.get("weight_decay", 0.0),
                group.get("eps", 1e-10),
            )
        )

    def step(self, closure: Any = None) -> Any:
        """Perform a single Adagrad step."""
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

    def _append_engine_optim(self, group: dict[str, Any]) -> None:
        self._engine_optims.append(
            _C_engine.Adadelta(
                [_unwrap(p) for p in group["params"]],
                group["lr"],
                group.get("rho", 0.9),
                group.get("eps", 1e-6),
                group.get("weight_decay", 0.0),
            )
        )

    def step(self, closure: Any = None) -> Any:
        """Perform a single Adadelta step."""
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

    def _append_engine_optim(self, group: dict[str, Any]) -> None:
        self._engine_optims.append(
            _C_engine.Adamax(
                [_unwrap(p) for p in group["params"]],
                group["lr"],
                group.get("beta1", 0.9),
                group.get("beta2", 0.999),
                group.get("eps", 1e-8),
                group.get("weight_decay", 0.0),
            )
        )

    def step(self, closure: Any = None) -> Any:
        """Perform a single Adamax step."""
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

    def _append_engine_optim(self, group: dict[str, Any]) -> None:
        self._engine_optims.append(
            _C_engine.RAdam(
                [_unwrap(p) for p in group["params"]],
                group["lr"],
                group.get("beta1", 0.9),
                group.get("beta2", 0.999),
                group.get("eps", 1e-8),
                group.get("weight_decay", 0.0),
            )
        )

    def step(self, closure: Any = None) -> Any:
        """Perform a single RAdam step."""
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

    def _append_engine_optim(self, group: dict[str, Any]) -> None:
        self._engine_optims.append(
            _C_engine.NAdam(
                [_unwrap(p) for p in group["params"]],
                group["lr"],
                group.get("beta1", 0.9),
                group.get("beta2", 0.999),
                group.get("eps", 1e-8),
                group.get("weight_decay", 0.0),
            )
        )

    def step(self, closure: Any = None) -> Any:
        """Perform a single NAdam step."""
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

    def _append_engine_optim(self, group: dict[str, Any]) -> None:
        self._engine_optims.append(
            _C_engine.ASGD(
                [_unwrap(p) for p in group["params"]],
                group["lr"],
                group.get("lambd", 1e-4),
                group.get("alpha", 0.75),
                group.get("t0", 1e6),
                group.get("weight_decay", 0.0),
            )
        )

    def step(self, closure: Any = None) -> Any:
        """Perform a single ASGD step."""
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

    def _append_engine_optim(self, group: dict[str, Any]) -> None:
        self._engine_optims.append(
            _C_engine.Rprop(
                [_unwrap(p) for p in group["params"]],
                group["lr"],
                group.get("eta_minus", 0.5),
                group.get("eta_plus", 1.2),
                group.get("step_min", 1e-6),
                group.get("step_max", 50.0),
            )
        )

    def step(self, closure: Any = None) -> Any:
        """Perform a single Rprop step."""
        loss = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()
        return loss
