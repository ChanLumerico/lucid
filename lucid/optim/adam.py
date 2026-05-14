"""
Adam and AdamW optimizers.
"""

from typing import Iterable, cast
from lucid._tensor.tensor import Tensor
from lucid._types import _OptimizerClosure
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap
from lucid.nn.parameter import Parameter
from lucid.optim.optimizer import Optimizer


class Adam(Optimizer):
    r"""Adaptive Moment Estimation optimizer (Kingma & Ba, 2015).

    Combines the benefits of two earlier adaptive methods — AdaGrad's
    per-parameter learning rates derived from the running history of
    gradients, and RMSProp's exponential moving average of squared
    gradients — by maintaining two moment estimates and applying
    **bias correction** to compensate for their zero initialisation.
    The result is a near-parameterless optimiser that works well across
    a remarkably wide range of architectures and is the de-facto default
    for deep learning training.

    Parameters
    ----------
    params : iterable of Parameter
        Iterable of parameters to optimise, or dicts defining parameter
        groups with their own per-group hyperparameters.
    lr : float, optional
        Learning rate :math:`\alpha` (default: ``1e-3``).
    betas : tuple of float, optional
        Decay rates :math:`(\beta_1, \beta_2)` for the first and second
        moment running averages (default: ``(0.9, 0.999)``).
    eps : float, optional
        Term :math:`\varepsilon` added to the denominator for numerical
        stability (default: ``1e-8``).
    weight_decay : float, optional
        :math:`L_2` penalty coefficient.  Note that Adam folds this
        directly into the gradient before the moment update, which
        couples it with the adaptive learning rate; prefer
        :class:`AdamW` for properly decoupled weight decay.
    amsgrad : bool, optional
        Whether to use the AMSGrad variant (Reddi et al., 2018), which
        keeps the running max of the second moment to guarantee
        convergence under non-convex settings (default: ``False``).

    Notes
    -----
    The update rule for parameter :math:`\theta` with gradient
    :math:`g_t` at step :math:`t` is:

    .. math::

        m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
        v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
        \hat{m}_t &= m_t / (1 - \beta_1^t) \\
        \hat{v}_t &= v_t / (1 - \beta_2^t) \\
        \theta_t &= \theta_{t-1} - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \varepsilon)

    where :math:`m_t` is the running first moment (mean of gradients),
    :math:`v_t` is the running uncentered second moment (mean of squared
    gradients), and the hat-quantities apply bias correction so that
    :math:`\mathbb{E}[\hat{m}_t] = \mathbb{E}[g_t]` even at small
    :math:`t`.  The effective per-parameter learning rate is
    :math:`\alpha / (\sqrt{\hat{v}_t} + \varepsilon)` — small for
    high-variance gradients, large for stable ones.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.Adam(model.parameters(), lr=1e-3)
    >>> for x, y in dataloader:
    ...     optimizer.zero_grad()
    ...     loss = loss_fn(model(x), y)
    ...     loss.backward()
    ...     optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
    ) -> None:
        """Initialise the Adam.  See the class docstring for parameter semantics."""
        defaults: dict[str, object] = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        from lucid.nn.parameter import Parameter as _P

        params_list: list[_P] = group["params"]  # type: ignore[assignment]
        self._engine_optims.append(
            _C_engine.Adam(
                [_unwrap(p) for p in params_list],
                cast(float, group["lr"]),
                cast(float, group.get("beta1", 0.9)),
                cast(float, group.get("beta2", 0.999)),
                cast(float, group.get("eps", 1e-8)),
                cast(float, group.get("weight_decay", 0.0)),
            )
        )

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single Adam optimisation step.

        Calls the engine-level Adam update for each parameter group, which
        applies the bias-corrected first- and second-moment update rule.

        Parameters
        ----------
        closure : callable, optional
            A closure that re-evaluates the model and returns the loss.
            If provided, it is called **before** the parameter update and
            its return value is passed back to the caller.

        Returns
        -------
        Tensor or None
            The loss returned by ``closure``, or ``None`` if no closure
            was provided.

        Examples
        --------
        >>> optimizer.zero_grad()
        >>> loss = model(inputs)
        >>> loss.backward()
        >>> optimizer.step()
        """
        loss: Tensor | None = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()  # type: ignore[attr-defined]  # _EngineOptimizer.step() correct at runtime
        return loss


class AdamW(Optimizer):
    r"""Adam optimizer with decoupled weight decay regularisation.

    ``AdamW`` fixes the weight-decay coupling present in standard Adam by
    applying the decay directly to the parameters rather than adding it to
    the gradient.  The update rule is:

    .. math::

        m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
        v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
        \hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
        \hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
        \theta_t &= \theta_{t-1}
            - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
            - \alpha \lambda \theta_{t-1}

    The final term :math:`-\alpha \lambda \theta_{t-1}` is the decoupled
    weight decay; it is applied after the adaptive gradient step, not mixed
    into :math:`g_t`.

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise, or a list of parameter-group dicts.
    lr : float, optional
        Learning rate :math:`\alpha` (default: ``1e-3``).
    betas : tuple of float, optional
        Coefficients :math:`(\beta_1, \beta_2)` for computing running
        averages of the gradient and its square (default: ``(0.9, 0.999)``).
    eps : float, optional
        Term :math:`\epsilon` added to the denominator for numerical
        stability (default: ``1e-8``).
    weight_decay : float, optional
        Decoupled weight decay coefficient :math:`\lambda`
        (default: ``1e-2``).
    amsgrad : bool, optional
        Whether to use the AMSGrad variant that maintains the maximum of
        past squared gradients (default: ``False``).

    Attributes
    ----------
    param_groups : list of dict
        Parameter groups, each containing ``"params"``, ``"lr"``,
        ``"beta1"``, ``"beta2"``, ``"eps"``, ``"weight_decay"``, and
        ``"amsgrad"``.
    defaults : dict
        Default hyperparameter values.

    Notes
    -----
    Decoupled weight decay makes the effective regularisation independent
    of the learning rate, which simplifies hyperparameter tuning.
    ``AdamW`` is the recommended default optimizer for transformer-based
    models and generally outperforms ``Adam`` with L2 regularisation.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.AdamW(
    ...     model.parameters(), lr=1e-4, weight_decay=1e-2
    ... )
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
    ) -> None:
        """Initialise the AdamW.  See the class docstring for parameter semantics."""
        defaults: dict[str, object] = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        from lucid.nn.parameter import Parameter as _P

        params_list: list[_P] = group["params"]  # type: ignore[assignment]
        self._engine_optims.append(
            _C_engine.AdamW(
                [_unwrap(p) for p in params_list],
                cast(float, group["lr"]),
                cast(float, group.get("beta1", 0.9)),
                cast(float, group.get("beta2", 0.999)),
                cast(float, group.get("eps", 1e-8)),
                cast(float, group.get("weight_decay", 1e-2)),
            )
        )

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single AdamW optimisation step.

        Calls the engine-level AdamW update for each parameter group, which
        applies the adaptive gradient update followed by decoupled weight
        decay directly on the parameters.

        Parameters
        ----------
        closure : callable, optional
            A closure that re-evaluates the model and returns the loss.
            If provided, it is called **before** the parameter update and
            its return value is passed back to the caller.

        Returns
        -------
        Tensor or None
            The loss returned by ``closure``, or ``None`` if no closure
            was provided.

        Examples
        --------
        >>> optimizer.zero_grad()
        >>> loss = model(inputs)
        >>> loss.backward()
        >>> optimizer.step()
        """
        loss: Tensor | None = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()  # type: ignore[attr-defined]  # _EngineOptimizer.step() correct at runtime
        return loss
