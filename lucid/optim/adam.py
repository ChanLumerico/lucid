"""
Adam and AdamW optimizers.
"""

from typing import Any
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap
from lucid.optim.optimizer import Optimizer


class Adam(Optimizer):
    """
    Adaptive moment estimation optimizer.

    Args:
        params:       iterable of Parameters
        lr:           learning rate (default: 1e-3)
        betas:        (beta1, beta2) coefficients (default: (0.9, 0.999))
        eps:          numerical stability term (default: 1e-8)
        weight_decay: L2 penalty (default: 0)
        amsgrad:      whether to use AMSGrad variant (default: False)
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr, beta1=betas[0], beta2=betas[1],
            eps=eps, weight_decay=weight_decay, amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, Any]) -> None:
        self._engine_optims.append(
            _C_engine.Adam(
                [_unwrap(p) for p in group["params"]],
                group["lr"],
                group.get("beta1", 0.9),
                group.get("beta2", 0.999),
                group.get("eps", 1e-8),
                group.get("weight_decay", 0.0),
                group.get("amsgrad", False),
            )
        )

    def step(self, closure: Any = None) -> Any:
        """Perform a single Adam step."""
        loss = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()
        return loss


class AdamW(Optimizer):
    """
    Adam with decoupled weight decay.

    Args:
        params:       iterable of Parameters
        lr:           learning rate (default: 1e-3)
        betas:        (beta1, beta2) coefficients (default: (0.9, 0.999))
        eps:          numerical stability term (default: 1e-8)
        weight_decay: decoupled weight decay (default: 1e-2)
        amsgrad:      whether to use AMSGrad variant (default: False)
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr, beta1=betas[0], beta2=betas[1],
            eps=eps, weight_decay=weight_decay, amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, Any]) -> None:
        self._engine_optims.append(
            _C_engine.AdamW(
                [_unwrap(p) for p in group["params"]],
                group["lr"],
                group.get("beta1", 0.9),
                group.get("beta2", 0.999),
                group.get("eps", 1e-8),
                group.get("weight_decay", 1e-2),
            )
        )

    def step(self, closure: Any = None) -> Any:
        """Perform a single AdamW step."""
        loss = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()
        return loss
