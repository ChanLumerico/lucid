"""
Adam and AdamW optimizers.
"""

from lucid._tensor.tensor import Tensor
from lucid._types import _OptimizerClosure
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap
from lucid.optim.optimizer import Optimizer


class Adam(Optimizer):
    """Adaptive Moment Estimation optimizer.

    Maintains per-parameter first and second moment estimates to adapt
    the learning rate. See :cite:t:`Kingma2015Adam`.

    Parameters
    ----------
    params : iterable of Parameter
        Iterable of parameters to optimize or dicts defining parameter groups.
    lr : float, optional
        Learning rate (default: 1e-3).
    betas : tuple of float, optional
        Coefficients for computing running averages of gradient and its square
        (default: ``(0.9, 0.999)``).
    eps : float, optional
        Term added to the denominator for numerical stability (default: 1e-8).
    weight_decay : float, optional
        L2 penalty coefficient (default: 0).
    amsgrad : bool, optional
        Whether to use the AMSGrad variant (default: ``False``).

    Examples
    --------
    >>> optimizer = optim.Adam(model.parameters(), lr=1e-3)
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
    """

    def __init__(
        self,
        params: object,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
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

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single Adam step."""
        loss: Tensor | None = closure() if closure is not None else None
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
        params: object,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
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

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single AdamW step."""
        loss: Tensor | None = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()
        return loss
