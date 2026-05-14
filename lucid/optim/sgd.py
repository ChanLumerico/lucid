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
    r"""Stochastic Gradient Descent optimizer with optional momentum and weight decay.

    Implements the classic SGD update rule.  Without momentum the update is:

    .. math::

        \theta_{t+1} = \theta_t - \alpha \, \nabla L(\theta_t)

    With momentum (Polyak momentum), a velocity buffer :math:`v` is
    maintained and the update becomes:

    .. math::

        v_{t+1} &= \mu \, v_t + (1 - \tau) \, \nabla L(\theta_t) \\
        \theta_{t+1} &= \theta_t - \alpha \, v_{t+1}

    where :math:`\mu` is the momentum factor and :math:`\tau` is the
    dampening coefficient.  With Nesterov momentum the gradient is
    evaluated at the *lookahead* position:

    .. math::

        \theta_{t+1} = \theta_t - \alpha
            \bigl(\nabla L(\theta_t) + \mu \, v_{t+1}\bigr)

    L2 weight decay adds :math:`\lambda \theta_t` to the gradient before
    the momentum step:

    .. math::

        g_t = \nabla L(\theta_t) + \lambda \, \theta_t

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise, or a list of parameter-group dicts.
    lr : float
        Learning rate :math:`\alpha`.
    momentum : float, optional
        Momentum factor :math:`\mu` (default: ``0``).  Set to a value
        such as ``0.9`` to enable momentum.
    dampening : float, optional
        Dampening factor :math:`\tau` for the momentum buffer
        (default: ``0``).  Has no effect when ``momentum=0``.
    weight_decay : float, optional
        L2 regularisation coefficient :math:`\lambda` (default: ``0``).
    nesterov : bool, optional
        If ``True``, use Nesterov momentum (default: ``False``).
        Requires ``momentum > 0`` and ``dampening == 0``.

    Attributes
    ----------
    param_groups : list of dict
        Parameter groups, each containing ``"params"``, ``"lr"``,
        ``"momentum"``, ``"dampening"``, ``"weight_decay"``, and
        ``"nesterov"``.
    defaults : dict
        Default hyperparameter values.

    Notes
    -----
    SGD with momentum is the de-facto standard for training image
    classifiers.  Nesterov momentum often converges faster than vanilla
    momentum because it incorporates a correction based on where the
    parameters will be after the momentum step.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(
    ...     model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
    ... )
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
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
        """Initialise the SGD.  See the class docstring for parameter semantics."""
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
