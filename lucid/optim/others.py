"""
Additional optimizers: RMSprop, Adagrad, Adadelta, Adamax, RAdam, NAdam, ASGD, Rprop.
"""

from typing import Iterable, cast
from lucid._tensor.tensor import Tensor
from lucid._types import _OptimizerClosure
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap
from lucid.nn.parameter import Parameter
from lucid.optim.optimizer import Optimizer


class RMSprop(Optimizer):
    r"""Root Mean Square Propagation optimizer.

    RMSprop maintains a running average of the squared gradient for each
    parameter and divides the gradient by the square root of that average,
    which normalises the effective learning rate per parameter:

    .. math::

        v_t &= \alpha \, v_{t-1} + (1 - \alpha) \, g_t^2 \\
        \theta_t &= \theta_{t-1}
            - \frac{\eta}{\sqrt{v_t} + \epsilon} \, g_t

    With momentum an additional velocity buffer :math:`b` is maintained:

    .. math::

        b_t &= \mu \, b_{t-1}
            + \frac{\eta}{\sqrt{v_t} + \epsilon} \, g_t \\
        \theta_t &= \theta_{t-1} - b_t

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise, or a list of parameter-group dicts.
    lr : float, optional
        Learning rate :math:`\eta` (default: ``1e-2``).
    alpha : float, optional
        Smoothing constant :math:`\alpha` for the running squared-gradient
        average (default: ``0.99``).
    eps : float, optional
        Term :math:`\epsilon` added to the denominator for numerical
        stability (default: ``1e-8``).
    weight_decay : float, optional
        L2 regularisation coefficient (default: ``0``).
    momentum : float, optional
        Momentum factor :math:`\mu` (default: ``0``).
    centered : bool, optional
        If ``True``, normalise by the estimated variance rather than the
        raw second moment (default: ``False``).

    Attributes
    ----------
    param_groups : list of dict
        Parameter groups with keys ``"params"``, ``"lr"``, ``"alpha"``,
        ``"eps"``, ``"weight_decay"``, ``"momentum"``, and ``"centered"``.
    defaults : dict
        Default hyperparameter values.

    Notes
    -----
    RMSprop was proposed as an unpublished improvement for non-stationary
    objectives and recurrent networks.  It works well with a moderate
    learning rate and is often used for reinforcement learning tasks.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99)
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
    ) -> None:
        """Initialise the RMSprop.  See the class docstring for parameter semantics."""
        defaults: dict[str, object] = dict(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engine_optims.append(
            _C_engine.RMSprop(
                [_unwrap(p) for p in group["params"]],  # type: ignore[attr-defined]
                cast(float, group["lr"]),
                cast(float, group.get("alpha", 0.99)),
                cast(float, group.get("eps", 1e-8)),
                cast(float, group.get("weight_decay", 0.0)),
                cast(float, group.get("momentum", 0.0)),
            )
        )

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single RMSprop step."""
        loss: Tensor | None = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()  # type: ignore[attr-defined]  # _EngineOptimizer.step() is correct at runtime
        return loss


class Adagrad(Optimizer):
    r"""Adaptive Gradient optimizer.

    Adagrad adapts the learning rate for each parameter based on the sum
    of all past squared gradients.  Parameters that receive large or
    frequent gradient updates get smaller effective learning rates:

    .. math::

        G_t &= G_{t-1} + g_t^2 \\
        \theta_t &= \theta_{t-1}
            - \frac{\eta_t}{\sqrt{G_t} + \epsilon} \, g_t

    where the effective learning rate decays with time as:

    .. math::

        \eta_t = \frac{\eta_0}{1 + (t - 1) \cdot \lambda}

    and :math:`\lambda` is the ``lr_decay`` parameter.

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise, or a list of parameter-group dicts.
    lr : float, optional
        Initial learning rate :math:`\eta_0` (default: ``1e-2``).
    lr_decay : float, optional
        Learning-rate decay applied to the effective LR at each step
        :math:`\lambda` (default: ``0``).
    weight_decay : float, optional
        L2 regularisation coefficient (default: ``0``).
    eps : float, optional
        Term :math:`\epsilon` added to the denominator for numerical
        stability (default: ``1e-10``).

    Attributes
    ----------
    param_groups : list of dict
        Parameter groups with keys ``"params"``, ``"lr"``, ``"lr_decay"``,
        ``"weight_decay"``, and ``"eps"``.
    defaults : dict
        Default hyperparameter values.

    Notes
    -----
    Adagrad is particularly well-suited for sparse data (e.g. NLP with
    word embeddings) because infrequently updated parameters retain a
    larger effective learning rate.  The main drawback is that the
    accumulated squared-gradient sum :math:`G_t` only grows, so the
    effective learning rate can become vanishingly small over long training
    runs.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.Adagrad(model.parameters(), lr=1e-2)
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1e-2,
        lr_decay: float = 0,
        weight_decay: float = 0,
        eps: float = 1e-10,
    ) -> None:
        """Initialise the Adagrad.  See the class docstring for parameter semantics."""
        defaults: dict[str, object] = dict(
            lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, eps=eps
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engine_optims.append(
            _C_engine.Adagrad(
                [_unwrap(p) for p in group["params"]],  # type: ignore[attr-defined]
                cast(float, group["lr"]),
                cast(float, group.get("lr_decay", 0.0)),
                cast(float, group.get("weight_decay", 0.0)),
                cast(float, group.get("eps", 1e-10)),
            )
        )

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single Adagrad step."""
        loss: Tensor | None = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()  # type: ignore[attr-defined]  # _EngineOptimizer.step() is correct at runtime
        return loss


class Adadelta(Optimizer):
    r"""Adadelta optimizer — an adaptive learning rate method with no global LR.

    Adadelta addresses Adagrad's aggressive, monotonically decreasing learning
    rate by limiting the accumulated past gradients to a fixed-size window via
    exponential moving averages.  No global learning rate is required in the
    canonical form:

    .. math::

        E[g^2]_t &= \rho \, E[g^2]_{t-1} + (1 - \rho) \, g_t^2 \\
        \Delta\theta_t &=
            -\frac{\sqrt{E[\Delta\theta^2]_{t-1} + \epsilon}}
                   {\sqrt{E[g^2]_t + \epsilon}} \, g_t \\
        E[\Delta\theta^2]_t &= \rho \, E[\Delta\theta^2]_{t-1}
            + (1 - \rho) \, \Delta\theta_t^2 \\
        \theta_t &= \theta_{t-1} + \eta \, \Delta\theta_t

    where :math:`\eta` is ``lr`` (defaults to ``1.0`` in the original
    formulation) and :math:`\rho` controls the decay window size.

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise, or a list of parameter-group dicts.
    lr : float, optional
        Scaling factor :math:`\eta` applied to the update (default: ``1.0``).
    rho : float, optional
        Coefficient for the running averages of squared gradients and
        squared updates :math:`\rho` (default: ``0.9``).
    eps : float, optional
        Term :math:`\epsilon` added to the denominator for numerical
        stability (default: ``1e-6``).
    weight_decay : float, optional
        L2 regularisation coefficient (default: ``0``).

    Attributes
    ----------
    param_groups : list of dict
        Parameter groups with keys ``"params"``, ``"lr"``, ``"rho"``,
        ``"eps"``, and ``"weight_decay"``.
    defaults : dict
        Default hyperparameter values.

    Notes
    -----
    Because Adadelta automatically adapts its learning rate based on a
    window of recent gradient magnitudes, it is relatively robust to
    hyperparameter choices and does not require manual LR tuning.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.Adadelta(model.parameters(), rho=0.9, eps=1e-6)
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0,
    ) -> None:
        """Initialise the Adadelta.  See the class docstring for parameter semantics."""
        defaults: dict[str, object] = dict(
            lr=lr, rho=rho, eps=eps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engine_optims.append(
            _C_engine.Adadelta(
                [_unwrap(p) for p in group["params"]],  # type: ignore[attr-defined]
                cast(float, group["lr"]),
                cast(float, group.get("rho", 0.9)),
                cast(float, group.get("eps", 1e-6)),
                cast(float, group.get("weight_decay", 0.0)),
            )
        )

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single Adadelta step."""
        loss: Tensor | None = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()  # type: ignore[attr-defined]  # _EngineOptimizer.step() is correct at runtime
        return loss


class Adamax(Optimizer):
    r"""Adamax optimizer — a variant of Adam based on the infinity norm.

    Adamax generalises Adam by using the :math:`\ell_\infty` norm instead
    of the :math:`\ell_2` norm for the second-moment estimate.  The update
    rule replaces :math:`v_t` with the element-wise maximum of past
    absolute gradients scaled by :math:`\beta_2`:

    .. math::

        m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
        u_t &= \max\!\left(\beta_2 \, u_{t-1},\; |g_t|\right) \\
        \theta_t &= \theta_{t-1}
            - \frac{\eta}{(1 - \beta_1^t)} \cdot \frac{m_t}{u_t + \epsilon}

    Because :math:`u_t` is bounded by :math:`\max_k \beta_2^k |g_{t-k}|`,
    the effective step size is naturally bounded.

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise, or a list of parameter-group dicts.
    lr : float, optional
        Learning rate :math:`\eta` (default: ``2e-3``).
    betas : tuple of float, optional
        Coefficients :math:`(\beta_1, \beta_2)` for the first-moment
        estimate and the :math:`\ell_\infty` norm decay
        (default: ``(0.9, 0.999)``).
    eps : float, optional
        Term :math:`\epsilon` added to the denominator for numerical
        stability (default: ``1e-8``).
    weight_decay : float, optional
        L2 regularisation coefficient (default: ``0``).

    Attributes
    ----------
    param_groups : list of dict
        Parameter groups with keys ``"params"``, ``"lr"``, ``"beta1"``,
        ``"beta2"``, ``"eps"``, and ``"weight_decay"``.
    defaults : dict
        Default hyperparameter values.

    Notes
    -----
    Adamax can be more stable than Adam on problems where gradients are
    sparse or have large outliers, because the infinity norm is less
    sensitive to large individual gradient magnitudes than the L2 norm.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.Adamax(model.parameters(), lr=2e-3)
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 2e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
        """Initialise the Adamax.  See the class docstring for parameter semantics."""
        defaults: dict[str, object] = dict(
            lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engine_optims.append(
            _C_engine.Adamax(
                [_unwrap(p) for p in group["params"]],  # type: ignore[attr-defined]
                cast(float, group["lr"]),
                cast(float, group.get("beta1", 0.9)),
                cast(float, group.get("beta2", 0.999)),
                cast(float, group.get("eps", 1e-8)),
                cast(float, group.get("weight_decay", 0.0)),
            )
        )

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single Adamax step."""
        loss: Tensor | None = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()  # type: ignore[attr-defined]  # _EngineOptimizer.step() is correct at runtime
        return loss


class RAdam(Optimizer):
    r"""Rectified Adam optimizer with variance-adapted step size.

    RAdam addresses the large variance in Adam's effective learning rate
    during the early training steps (when the moving averages are poorly
    initialised) by computing a rectification term that smoothly transitions
    between SGD and Adam.

    The maximum length of the approximated SMA is:

    .. math::

        \rho_\infty = \frac{2}{1 - \beta_2} - 1

    At step :math:`t` the current SMA length estimate is:

    .. math::

        \rho_t = \rho_\infty
            - \frac{2 t \, \beta_2^t}{1 - \beta_2^t}

    When :math:`\rho_t > 4` the variance is tractable and a rectified
    adaptive step is used:

    .. math::

        r_t &= \sqrt{
            \frac{(\rho_t - 4)(\rho_t - 2)\rho_\infty}
                 {(\rho_\infty - 4)(\rho_\infty - 2)\rho_t}} \\
        \theta_t &= \theta_{t-1}
            - \alpha \, r_t
              \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}

    Otherwise the update falls back to SGD with bias-corrected momentum.

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise, or a list of parameter-group dicts.
    lr : float, optional
        Learning rate :math:`\alpha` (default: ``1e-3``).
    betas : tuple of float, optional
        Coefficients :math:`(\beta_1, \beta_2)` for the first- and
        second-moment estimates (default: ``(0.9, 0.999)``).
    eps : float, optional
        Term :math:`\epsilon` for numerical stability (default: ``1e-8``).
    weight_decay : float, optional
        L2 regularisation coefficient (default: ``0``).

    Attributes
    ----------
    param_groups : list of dict
        Parameter groups with keys ``"params"``, ``"lr"``, ``"beta1"``,
        ``"beta2"``, ``"eps"``, and ``"weight_decay"``.
    defaults : dict
        Default hyperparameter values.

    Notes
    -----
    RAdam removes the need for a warmup schedule by automatically
    stabilising the adaptive learning rate in the early stages of training.
    It is a drop-in replacement for Adam that is less sensitive to the
    choice of learning rate.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.RAdam(model.parameters(), lr=1e-3)
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
        weight_decay: float = 0,
    ) -> None:
        """Initialise the RAdam.  See the class docstring for parameter semantics."""
        defaults: dict[str, object] = dict(
            lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engine_optims.append(
            _C_engine.RAdam(
                [_unwrap(p) for p in group["params"]],  # type: ignore[attr-defined]
                cast(float, group["lr"]),
                cast(float, group.get("beta1", 0.9)),
                cast(float, group.get("beta2", 0.999)),
                cast(float, group.get("eps", 1e-8)),
                cast(float, group.get("weight_decay", 0.0)),
            )
        )

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single RAdam step."""
        loss: Tensor | None = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()  # type: ignore[attr-defined]  # _EngineOptimizer.step() is correct at runtime
        return loss


class NAdam(Optimizer):
    r"""Nesterov-accelerated Adaptive Moment Estimation optimizer.

    NAdam incorporates Nesterov momentum into Adam by replacing the
    standard first-moment estimate :math:`\hat{m}_t` in the denominator
    with a one-step lookahead estimate.  The update rule is:

    .. math::

        m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
        v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
        \hat{m}_t^{\text{Nesterov}}
            &= \frac{\beta_1 m_t}{1 - \beta_1^{t+1}}
             + \frac{(1 - \beta_1) g_t}{1 - \beta_1^t} \\
        \theta_t &= \theta_{t-1}
            - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon}
              \hat{m}_t^{\text{Nesterov}}

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise, or a list of parameter-group dicts.
    lr : float, optional
        Learning rate :math:`\alpha` (default: ``2e-3``).
    betas : tuple of float, optional
        Coefficients :math:`(\beta_1, \beta_2)` for the first- and
        second-moment estimates (default: ``(0.9, 0.999)``).
    eps : float, optional
        Term :math:`\epsilon` for numerical stability (default: ``1e-8``).
    weight_decay : float, optional
        L2 regularisation coefficient (default: ``0``).

    Attributes
    ----------
    param_groups : list of dict
        Parameter groups with keys ``"params"``, ``"lr"``, ``"beta1"``,
        ``"beta2"``, ``"eps"``, and ``"weight_decay"``.
    defaults : dict
        Default hyperparameter values.

    Notes
    -----
    NAdam often converges faster than vanilla Adam because the Nesterov
    lookahead provides a more accurate gradient direction.  It is
    particularly effective on recurrent networks and tasks with noisy
    gradients.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.NAdam(model.parameters(), lr=2e-3)
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 2e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
        """Initialise the NAdam.  See the class docstring for parameter semantics."""
        defaults: dict[str, object] = dict(
            lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engine_optims.append(
            _C_engine.NAdam(
                [_unwrap(p) for p in group["params"]],  # type: ignore[attr-defined]
                cast(float, group["lr"]),
                cast(float, group.get("beta1", 0.9)),
                cast(float, group.get("beta2", 0.999)),
                cast(float, group.get("eps", 1e-8)),
                cast(float, group.get("weight_decay", 0.0)),
            )
        )

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single NAdam step."""
        loss: Tensor | None = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()  # type: ignore[attr-defined]  # _EngineOptimizer.step() is correct at runtime
        return loss


class ASGD(Optimizer):
    r"""Averaged Stochastic Gradient Descent optimizer.

    ASGD performs standard SGD updates but maintains a running average of
    the iterate sequence, which serves as the final parameter estimate.
    The averaging improves convergence in the presence of noise and is
    particularly effective near the end of training.

    The SGD update with L2 regularisation is:

    .. math::

        \theta_t = \theta_{t-1}
            - \eta_t \bigl(g_t + \lambda \, \theta_{t-1}\bigr)

    where the effective learning rate decays as:

    .. math::

        \eta_t = \frac{\eta_0}{(1 + \lambda \, \eta_0 \, t)^\alpha}

    The Polyak–Ruppert average is then:

    .. math::

        \bar{\theta}_t = \frac{1}{t - t_0} \sum_{k=t_0}^{t} \theta_k
        \quad \text{for } t \ge t_0

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise, or a list of parameter-group dicts.
    lr : float, optional
        Initial learning rate :math:`\eta_0` (default: ``1e-2``).
    lambd : float, optional
        Decay term :math:`\lambda` (default: ``1e-4``).
    alpha : float, optional
        Power for LR decay :math:`\alpha` (default: ``0.75``).
    t0 : float, optional
        Step at which averaging begins (default: ``1e6``).
    weight_decay : float, optional
        L2 regularisation coefficient (default: ``0``).

    Attributes
    ----------
    param_groups : list of dict
        Parameter groups with keys ``"params"``, ``"lr"``, ``"lambd"``,
        ``"alpha"``, ``"t0"``, and ``"weight_decay"``.
    defaults : dict
        Default hyperparameter values.

    Notes
    -----
    ASGD can match or exceed the convergence rate of SGD with careful
    learning-rate tuning, and the averaging step provides additional
    regularisation.  The default ``t0=1e6`` delays averaging until very
    late in training; reduce it to start averaging earlier.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.ASGD(model.parameters(), lr=1e-2, t0=1e5)
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1e-2,
        lambd: float = 1e-4,
        alpha: float = 0.75,
        t0: float = 1e6,
        weight_decay: float = 0,
    ) -> None:
        """Initialise the ASGD.  See the class docstring for parameter semantics."""
        defaults: dict[str, object] = dict(
            lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engine_optims.append(
            _C_engine.ASGD(
                [_unwrap(p) for p in group["params"]],  # type: ignore[attr-defined]
                cast(float, group["lr"]),
                cast(float, group.get("lambd", 1e-4)),
                cast(float, group.get("alpha", 0.75)),
                cast(float, group.get("t0", 1e6)),
                cast(float, group.get("weight_decay", 0.0)),
            )
        )

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single ASGD step."""
        loss: Tensor | None = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()  # type: ignore[attr-defined]  # _EngineOptimizer.step() is correct at runtime
        return loss


class Rprop(Optimizer):
    r"""Resilient Backpropagation optimizer.

    Rprop ignores gradient magnitudes and adapts a per-parameter step size
    based only on the **sign** of the gradient.  If the sign of the
    gradient does not change between steps the step size is increased; if
    the sign reverses the step size is decreased:

    .. math::

        \Delta_{i,t} =
        \begin{cases}
            \min(\Delta_{i,t-1} \cdot \eta^+,\; \Delta_{\max})
                & \text{if } g_{i,t} \cdot g_{i,t-1} > 0 \\
            \max(\Delta_{i,t-1} \cdot \eta^-,\; \Delta_{\min})
                & \text{if } g_{i,t} \cdot g_{i,t-1} < 0 \\
            \Delta_{i,t-1} & \text{otherwise}
        \end{cases}

    The parameter update is then:

    .. math::

        \theta_{i,t} = \theta_{i,t-1}
            - \operatorname{sign}(g_{i,t}) \cdot \Delta_{i,t}

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise, or a list of parameter-group dicts.
    lr : float, optional
        Initial step size for each parameter (default: ``1e-2``).
    etas : tuple of float, optional
        Multiplicative decrease and increase factors
        :math:`(\eta^-, \eta^+)` (default: ``(0.5, 1.2)``).
    step_sizes : tuple of float, optional
        Minimum and maximum allowed step sizes
        :math:`(\Delta_{\min}, \Delta_{\max})` (default: ``(1e-6, 50)``).

    Attributes
    ----------
    param_groups : list of dict
        Parameter groups with keys ``"params"``, ``"lr"``, ``"eta_minus"``,
        ``"eta_plus"``, ``"step_min"``, and ``"step_max"``.
    defaults : dict
        Default hyperparameter values.

    Notes
    -----
    Rprop is particularly effective for full-batch training because the
    sign-based update is well-defined when gradients are deterministic.
    For stochastic mini-batch training RMSprop or Adam are generally
    preferred.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.Rprop(model.parameters(), lr=1e-2, etas=(0.5, 1.2))
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1e-2,
        etas: tuple[float, float] = (0.5, 1.2),
        step_sizes: tuple[float, float] = (1e-6, 50),
    ) -> None:
        """Initialise the Rprop.  See the class docstring for parameter semantics."""
        defaults: dict[str, object] = dict(
            lr=lr,
            eta_minus=etas[0],
            eta_plus=etas[1],
            step_min=step_sizes[0],
            step_max=step_sizes[1],
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engine_optims.append(
            _C_engine.Rprop(
                [_unwrap(p) for p in group["params"]],  # type: ignore[attr-defined]
                cast(float, group["lr"]),
                cast(float, group.get("eta_minus", 0.5)),
                cast(float, group.get("eta_plus", 1.2)),
                cast(float, group.get("step_min", 1e-6)),
                cast(float, group.get("step_max", 50.0)),
            )
        )

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single Rprop step."""
        loss: Tensor | None = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()  # type: ignore[attr-defined]  # _EngineOptimizer.step() is correct at runtime
        return loss


class SparseAdam(Optimizer):
    r"""Adam optimizer designed for sparse gradient workloads.

    Implements a dense Adam update that is API-compatible with the sparse
    variant used in embedding-heavy models.  All moment state is stored as
    dense :class:`~lucid._C.engine.TensorImpl` buffers and updated using
    engine operations, so the optimizer works correctly on both CPU and GPU.

    The update rule is identical to standard Adam:

    .. math::

        m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
        v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
        \hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
        \hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
        \theta_t &= \theta_{t-1}
            - \frac{\alpha \sqrt{1 - \beta_2^t}}{1 - \beta_1^t}
              \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}

    Moment buffers are allocated lazily on the first step for each
    parameter to avoid unnecessary memory usage for parameters with
    all-zero gradients.

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise, or a list of parameter-group dicts.
    lr : float, optional
        Learning rate :math:`\alpha` (default: ``1e-3``).
    betas : tuple of float, optional
        Coefficients :math:`(\beta_1, \beta_2)` for the first- and
        second-moment estimates (default: ``(0.9, 0.999)``).
    eps : float, optional
        Term :math:`\epsilon` for numerical stability (default: ``1e-8``).

    Attributes
    ----------
    param_groups : list of dict
        Parameter groups with keys ``"params"``, ``"lr"``, ``"betas"``,
        and ``"eps"``.
    defaults : dict
        Default hyperparameter values.

    Notes
    -----
    ``SparseAdam`` skips parameter updates when the gradient is ``None``,
    which is the common case for embedding rows that were not accessed in
    the current mini-batch.  This makes it efficient even though the
    underlying storage is dense.

    Unlike the engine-backed optimizers (Adam, AdamW, SGD), ``SparseAdam``
    manages its own Python-side moment buffers and does **not** use a C++
    engine optimizer object.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.SparseAdam(
    ...     model.embedding.parameters(), lr=1e-3
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
    ) -> None:
        """Initialise the SparseAdam.  See the class docstring for parameter semantics."""
        defaults: dict[str, object] = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        # Per-parameter moment state (TensorImpl, None before first step)
        self._step: list[int] = [0] * self._n_params()
        self._exp_avg: list[object] = [None] * self._n_params()
        self._exp_avg_sq: list[object] = [None] * self._n_params()

    def _n_params(self) -> int:
        return sum(len(g["params"]) for g in self.param_groups)  # type: ignore[arg-type, misc]

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single SparseAdam optimisation step.

        Iterates over all parameters.  For each parameter whose gradient
        is not ``None``, lazily initialises first- and second-moment
        buffers (on first call), increments the step counter, computes
        bias-corrected moment estimates, and applies the Adam update
        in-place via engine ops.

        Parameters with ``None`` gradients (e.g. embedding rows not
        accessed in the current batch) are skipped entirely.

        Parameters
        ----------
        closure : callable, optional
            A closure that re-evaluates the model and returns the loss.
            If provided, it is called **before** the parameter updates.

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
        loss = closure() if closure is not None else None
        flat_idx = 0
        for group in self.param_groups:
            lr = group["lr"]
            b1: float
            b2: float
            b1, b2 = group["betas"]  # type: ignore[misc]  # betas is tuple[float, float] at runtime
            eps = group["eps"]
            for p in group["params"]:  # type: ignore[attr-defined]  # params is list[Parameter] at runtime
                if p.grad is None:
                    flat_idx += 1
                    continue

                pi = p._impl
                gi = p.grad._impl
                dv = pi.device
                dt = pi.dtype
                sh = list(pi.shape)

                # Initialise moment buffers
                if self._exp_avg[flat_idx] is None:
                    self._exp_avg[flat_idx] = _C_engine.zeros(sh, dt, dv)
                    self._exp_avg_sq[flat_idx] = _C_engine.zeros(sh, dt, dv)

                self._step[flat_idx] += 1
                t = self._step[flat_idx]
                m = self._exp_avg[flat_idx]
                v = self._exp_avg_sq[flat_idx]

                def _scale(
                    tensor: _C_engine.TensorImpl, scalar: float
                ) -> _C_engine.TensorImpl:
                    return _C_engine.mul(tensor, _C_engine.full(sh, scalar, dt, dv))

                # m = b1 * m + (1 - b1) * g
                m = _C_engine.add(
                    _scale(cast(_C_engine.TensorImpl, m), b1),
                    _scale(cast(_C_engine.TensorImpl, gi), 1.0 - b1),
                )
                # v = b2 * v + (1 - b2) * g^2
                g_sq = _C_engine.mul(gi, gi)
                v = _C_engine.add(
                    _scale(cast(_C_engine.TensorImpl, v), b2), _scale(g_sq, 1.0 - b2)
                )

                self._exp_avg[flat_idx] = m
                self._exp_avg_sq[flat_idx] = v

                # Bias correction
                bc1 = 1.0 - b1**t
                bc2 = 1.0 - b2**t
                step_size = lr * (bc2**0.5) / bc1

                # p = p - step_size * m / (sqrt(v) + eps)
                denom = _C_engine.add(
                    _C_engine.sqrt(v), _C_engine.full(sh, eps, dt, dv)  # type: ignore[arg-type]
                )
                update = _scale(_C_engine.div(m, denom), step_size)
                new_p = _C_engine.sub(pi, update)
                pi.copy_from(new_p)
                flat_idx += 1
        return loss
