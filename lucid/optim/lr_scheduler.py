"""
Learning rate schedulers.
"""

import math
from typing import Callable
from lucid.optim.optimizer import Optimizer


class _LRScheduler:
    """Base class for all learning rate schedulers.

    Subclasses must implement :meth:`get_lr`, which returns a list of new
    learning rates — one per optimizer parameter group — given the current
    ``last_epoch`` counter.

    The scheduler tracks ``base_lrs``: the initial learning rates captured
    from the optimizer at construction time.  Every call to :meth:`step`
    increments ``last_epoch`` by one, recomputes the learning rates via
    :meth:`get_lr`, and writes them back into the optimizer's param groups.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer whose learning rate will be scheduled.
    last_epoch : int, optional
        The index of the last epoch.  Pass ``-1`` (default) to start from
        epoch 0 on the first :meth:`step` call; pass ``N`` to resume from
        an existing checkpoint where ``N`` steps have already been taken.
    verbose : bool, optional
        If ``True``, prints the updated learning rate to stdout after every
        :meth:`step` call (default: ``False``).

    Attributes
    ----------
    optimizer : Optimizer
        The optimizer being scheduled.
    base_lrs : list of float
        The initial learning rates of each param group captured at
        construction time.
    last_epoch : int
        Number of :meth:`step` calls completed so far.
    verbose : bool
        Whether to print learning-rate updates.

    Notes
    -----
    Schedulers should be stepped **after** the optimizer:

    .. code-block:: python

        for batch in dataloader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

    Examples
    --------
    This class is not instantiated directly.  Use a concrete subclass such
    as :class:`StepLR` or :class:`CosineAnnealingLR`.
    """

    def __init__(
        self, optimizer: Optimizer, last_epoch: int = -1, verbose: bool = False
    ) -> None:
        self.optimizer = optimizer
        self.last_epoch = (
            last_epoch + 1
        )  # starts at 0; after N step() calls, last_epoch == N
        self._step_count = 0
        self.verbose = verbose
        self.base_lrs: list[float] = [float(g["lr"]) for g in optimizer.param_groups]  # type: ignore[arg-type]  # lr is float at runtime
        self._last_lr: list[float] = list(self.base_lrs)

    def step(self) -> None:
        """Advance the scheduler by one epoch and update optimizer learning rates.

        Increments ``last_epoch`` by one, calls :meth:`get_lr` to obtain
        the new per-group learning rates, and writes them into the optimizer's
        param groups.  Also calls ``optimizer._sync_hyperparams()`` so that
        any engine-level optimizer mirrors the updated value.

        Should be called **once per epoch**, after ``optimizer.step()``.

        Examples
        --------
        >>> scheduler = lucid.optim.StepLR(optimizer, step_size=10)
        >>> for epoch in range(50):
        ...     train_one_epoch()
        ...     optimizer.step()
        ...     scheduler.step()
        """
        self.last_epoch += 1
        self._step_count += 1
        values = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, values):
            group["lr"] = lr
        self._last_lr = list(values)
        self.optimizer._sync_hyperparams()
        if self.verbose:
            self.print_lr(self.verbose, self.last_epoch, values)

    def get_lr(self) -> list[float]:
        """Compute new learning rates for all param groups.

        Returns
        -------
        list of float
            One learning rate per optimizer param group, based on the
            current ``last_epoch`` and ``base_lrs``.

        Raises
        ------
        NotImplementedError
            Concrete subclasses must override this method.
        """
        raise NotImplementedError

    def get_last_lr(self) -> list[float]:
        """Return the last learning rate computed by the scheduler.

        Returns
        -------
        list of float
            The current learning rate of each optimizer param group, read
            directly from the optimizer (not a cached copy).

        Examples
        --------
        >>> scheduler.step()
        >>> current_lrs = scheduler.get_last_lr()
        """
        return [float(g["lr"]) for g in self.optimizer.param_groups]  # type: ignore[arg-type]  # lr is float at runtime

    def print_lr(self, is_verbose: bool, epoch: int, lrs: list[float]) -> None:
        """Print the current learning rates to stdout when verbose mode is on.

        Parameters
        ----------
        is_verbose : bool
            When ``True``, the message is printed; when ``False``, this
            method is a no-op.
        epoch : int
            The epoch index to include in the printed message.
        lrs : list of float
            The new learning rates for each param group.
        """
        if is_verbose:
            for i, lr in enumerate(lrs):
                print(
                    f"Epoch {epoch}: adjusting learning rate of group {i} to {lr:.4e}."
                )


class StepLR(_LRScheduler):
    r"""Decay the learning rate by a fixed multiplicative factor every fixed number of epochs.

    At each epoch that is a multiple of ``step_size``, every param-group
    learning rate is multiplied by ``gamma``:

    .. math::

        \eta_t =
        \begin{cases}
            \eta_{t-1} \times \gamma & \text{if } t \bmod \text{step\_size} = 0 \\
            \eta_{t-1} & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    step_size : int
        Period (in epochs) between learning rate decays.
    gamma : float, optional
        Multiplicative factor applied at each decay step (default: ``0.1``).
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    step_size : int
        Decay period in epochs.
    gamma : float
        Multiplicative decay factor.

    Notes
    -----
    After ``k`` full decay steps the effective learning rate is:

    .. math::

        \eta_k = \eta_0 \cdot \gamma^{\lfloor t / \text{step\_size} \rfloor}

    A common choice is ``step_size=30, gamma=0.1`` for image classification
    tasks trained for 90 epochs.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    >>> scheduler = optim.StepLR(optimizer, step_size=30, gamma=0.1)
    >>> for epoch in range(90):
    ...     train(...)
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the StepLR.  See the class docstring for parameter semantics."""
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [float(g["lr"]) for g in self.optimizer.param_groups]  # type: ignore[arg-type]  # lr is float at runtime
        return [lr * self.gamma for lr in self.get_last_lr()]


class ExponentialLR(_LRScheduler):
    r"""Decay the learning rate by a constant factor every epoch.

    The learning rate is multiplied by ``gamma`` at every epoch, producing
    an exponential decay over time:

    .. math::

        \eta_t = \eta_0 \cdot \gamma^{t}

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    gamma : float
        Multiplicative factor applied each epoch.  Values in ``(0, 1)``
        produce decay; values greater than ``1`` produce growth (rarely
        useful).
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    gamma : float
        Multiplicative factor applied each epoch.

    Notes
    -----
    ``ExponentialLR`` applies the decay at every epoch without any
    plateau detection.  For a coarser schedule that decays only at
    certain milestones, prefer :class:`StepLR` or :class:`MultiStepLR`.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    >>> scheduler = optim.ExponentialLR(optimizer, gamma=0.95)
    >>> for epoch in range(100):
    ...     train(...)
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the ExponentialLR.  See the class docstring for parameter semantics."""
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        if self.last_epoch == 0:
            return self.base_lrs
        return [lr * self.gamma for lr in self.get_last_lr()]


class MultiStepLR(_LRScheduler):
    r"""Decay the learning rate by a fixed factor at a list of epoch milestones.

    Unlike :class:`StepLR`, the decay epochs need not be evenly spaced.
    At each epoch listed in ``milestones`` the learning rate is multiplied
    by ``gamma``:

    .. math::

        \eta_t = \eta_0 \cdot \gamma^{\sum_{m \in \text{milestones}} \mathbf{1}[t \ge m]}

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    milestones : list of int
        Sorted list of epoch indices at which the learning rate is decayed.
        If unsorted, the list is sorted internally.
    gamma : float, optional
        Multiplicative factor applied at each milestone (default: ``0.1``).
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    milestones : list of int
        Sorted epoch indices where decay is applied.
    gamma : float
        Multiplicative decay factor.

    Notes
    -----
    Useful when natural training phases occur at irregular intervals (e.g.,
    decay at epochs 50, 100, and 150 in a 200-epoch run).

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    >>> scheduler = optim.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)
    >>> for epoch in range(200):
    ...     train(...)
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: list[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the MultiStepLR.  See the class docstring for parameter semantics."""
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        if self.last_epoch not in self.milestones:
            return [float(g["lr"]) for g in self.optimizer.param_groups]  # type: ignore[arg-type]  # lr is float at runtime
        return [lr * self.gamma for lr in self.get_last_lr()]


class CosineAnnealingLR(_LRScheduler):
    r"""Anneal the learning rate following a cosine curve over ``T_max`` epochs.

    The learning rate at epoch :math:`t` is:

    .. math::

        \eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})
                  \left(1 + \cos\!\left(\frac{\pi\, t}{T_{\max}}\right)\right)

    where :math:`\eta_{\max}` is the initial learning rate captured from the
    optimizer (``base_lr``) and :math:`\eta_{\min}` is the floor set by
    ``eta_min``.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    T_max : int
        Half-period of the cosine cycle in epochs.  After ``T_max`` epochs
        the learning rate reaches ``eta_min``.
    eta_min : float, optional
        Minimum learning rate (default: ``0``).
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    T_max : int
        Decay period in epochs.
    eta_min : float
        Lower bound on the learning rate.

    Notes
    -----
    Cosine annealing produces a smooth, monotonically decreasing schedule
    that starts fast and slows near the minimum.  It is widely used for
    training deep networks and pairs naturally with warm-restarts
    (see :class:`CosineAnnealingWarmRestarts`).

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    >>> scheduler = optim.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    >>> for epoch in range(100):
    ...     train(...)
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the CosineAnnealingLR.  See the class docstring for parameter semantics."""
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
            for base_lr in self.base_lrs
        ]


class LambdaLR(_LRScheduler):
    r"""Set the learning rate using a user-supplied multiplicative factor function.

    At each epoch :math:`t`, the learning rate for param group :math:`i` is:

    .. math::

        \eta_t^{(i)} = \eta_0^{(i)} \cdot \lambda_i(t)

    where :math:`\eta_0^{(i)}` is the initial learning rate of group
    :math:`i` and :math:`\lambda_i` is the corresponding callable.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    lr_lambda : callable or list of callable
        A function ``f(epoch: int) -> float`` returning the multiplicative
        factor for the learning rate.  If a single callable is given it is
        applied to every param group; if a list is given it must have the
        same length as ``optimizer.param_groups``.
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    lr_lambdas : list of callable
        One factor function per param group.

    Notes
    -----
    ``LambdaLR`` provides maximum flexibility: any monotone or cyclic
    schedule can be encoded as a Python function.  For a linearly increasing
    warmup followed by constant LR:

    .. code-block:: python

        warmup = 5
        fn = lambda t: min(1.0, t / warmup)
        scheduler = optim.LambdaLR(optimizer, lr_lambda=fn)

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.Adam(model.parameters(), lr=1e-3)
    >>> scheduler = optim.LambdaLR(optimizer, lr_lambda=lambda t: 0.95 ** t)
    >>> for epoch in range(50):
    ...     train(...)
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable[[int], float] | list[Callable[[int], float]],
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the LambdaLR.  See the class docstring for parameter semantics."""
        if callable(lr_lambda):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            self.lr_lambdas = list(lr_lambda)
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        return [
            base_lr * fn(self.last_epoch)
            for base_lr, fn in zip(self.base_lrs, self.lr_lambdas)
        ]


class CyclicLR(_LRScheduler):
    r"""Cycle the learning rate between ``base_lr`` and ``max_lr``.

    Implements the triangular, triangular2, and exp_range cyclic policies.
    Within each cycle of length :math:`2 \times \text{step\_size\_up}` the
    learning rate rises linearly from ``base_lr`` to ``max_lr`` and then
    falls back:

    .. math::

        \text{cycle} &= \left\lfloor 1 + \frac{t}{2 \cdot s} \right\rfloor \\
        x &= \left|\frac{t}{s} - 2 \cdot \text{cycle} + 1\right| \\
        \text{scale} &= \max(0,\; 1 - x) \\
        \eta_t &= \eta_{\min} + (\eta_{\max} - \eta_{\min}) \cdot \text{scale}

    where :math:`s = \text{step\_size\_up}`.

    For ``mode="triangular2"`` the amplitude halves each cycle:

    .. math::

        \text{scale} \mathrel{/}= 2^{\text{cycle}-1}

    For ``mode="exp_range"`` the amplitude decays exponentially each step:

    .. math::

        \text{scale} \mathrel{\times}= \gamma^{t}

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    base_lr : float
        Lower boundary of the learning rate cycle.
    max_lr : float
        Upper boundary of the learning rate cycle.
    step_size_up : int, optional
        Number of steps in the increasing half of the cycle (default: ``2000``).
    mode : str, optional
        One of ``"triangular"`` (constant amplitude), ``"triangular2"``
        (amplitude halves each cycle), or ``"exp_range"`` (amplitude
        decays by :math:`\gamma^t` each step).  Default: ``"triangular"``.
    gamma : float, optional
        Decay factor used only in ``"exp_range"`` mode (default: ``1.0``).
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    base_lr_val : float
        Lower bound of the cycle.
    max_lr_val : float
        Upper bound of the cycle.
    step_size_up : int
        Half-cycle length in steps.
    mode : str
        Scaling policy name.
    gamma : float
        Decay factor for ``"exp_range"`` mode.

    Notes
    -----
    Cyclic learning rates can reduce the need for careful manual tuning by
    automatically exploring a range of rates.  Use ``step_size_up`` between
    2 and 10 times the number of iterations per epoch.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.01)
    >>> scheduler = optim.CyclicLR(
    ...     optimizer, base_lr=1e-4, max_lr=1e-2, step_size_up=500
    ... )
    >>> for batch in dataloader:
    ...     train_step(batch)
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float,
        max_lr: float,
        step_size_up: int = 2000,
        mode: str = "triangular",
        gamma: float = 1.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the CyclicLR.  See the class docstring for parameter semantics."""
        self.base_lr_val = base_lr
        self.max_lr_val = max_lr
        self.step_size_up = step_size_up
        self.mode = mode
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        cycle = math.floor(1 + self.last_epoch / (2 * self.step_size_up))
        x = abs(self.last_epoch / self.step_size_up - 2 * cycle + 1)
        scale = max(0.0, 1.0 - x)
        if self.mode == "triangular2":
            scale /= 2 ** (cycle - 1)
        elif self.mode == "exp_range":
            scale *= self.gamma**self.last_epoch
        lr = self.base_lr_val + (self.max_lr_val - self.base_lr_val) * scale
        return [lr] * len(self.optimizer.param_groups)


class ReduceLROnPlateau:
    r"""Reduce the learning rate when a monitored metric stops improving.

    Once the metric fails to improve by more than ``threshold`` for
    ``patience`` consecutive epochs, the learning rate is multiplied by
    ``factor``.  This is useful when progress stalls and a smaller step
    size may help escape a plateau:

    .. math::

        \eta \leftarrow \max(\eta \cdot \text{factor},\; \eta_{\min})

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    mode : str, optional
        One of ``"min"`` or ``"max"``.  In ``"min"`` mode the LR is
        reduced when the metric has stopped decreasing; in ``"max"`` mode
        it is reduced when the metric has stopped increasing
        (default: ``"min"``).
    factor : float, optional
        Multiplicative factor by which the learning rate is reduced
        (default: ``0.1``).  Must be less than 1.
    patience : int, optional
        Number of epochs with no improvement after which the LR is
        reduced (default: ``10``).
    verbose : bool, optional
        Print a message when the LR is reduced (default: ``False``).
    threshold : float, optional
        Minimum change to qualify as an improvement (default: ``1e-4``).
    min_lr : float, optional
        A lower bound on the learning rate (default: ``0``).

    Attributes
    ----------
    optimizer : Optimizer
        The optimizer being scheduled.
    mode : str
        ``"min"`` or ``"max"``.
    factor : float
        LR reduction factor.
    patience : int
        Epochs to wait before reducing.
    threshold : float
        Minimum improvement threshold.
    min_lr : float
        Floor on the learning rate.

    Notes
    -----
    Unlike the epoch-based schedulers that inherit from :class:`_LRScheduler`,
    ``ReduceLROnPlateau`` does **not** inherit from :class:`_LRScheduler`
    because it requires a metric value at each :meth:`step` call rather
    than advancing by a fixed epoch count.

    A typical use-case is to monitor validation loss:

    .. code-block:: python

        scheduler = optim.ReduceLROnPlateau(optimizer, mode="min", patience=5)
        for epoch in range(epochs):
            train(...)
            val_loss = validate(...)
            scheduler.step(val_loss)

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.Adam(model.parameters(), lr=1e-3)
    >>> scheduler = optim.ReduceLROnPlateau(
    ...     optimizer, mode="min", factor=0.5, patience=5
    ... )
    >>> for epoch in range(100):
    ...     val_loss = evaluate(...)
    ...     scheduler.step(val_loss)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        verbose: bool = False,
        threshold: float = 1e-4,
        min_lr: float = 0,
    ) -> None:
        """Initialise the ReduceLROnPlateau.  See the class docstring for parameter semantics."""
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.threshold = threshold
        self.min_lr = min_lr
        self._best: float = float("inf") if mode == "min" else float("-inf")
        self._num_bad_epochs = 0

    def step(self, metrics: float) -> None:
        """Update the scheduler with the latest monitored metric value.

        Checks whether the metric has improved relative to the stored best.
        If it has not improved for more than ``patience`` epochs, the
        learning rate is reduced by ``factor``.

        Parameters
        ----------
        metrics : float
            The current value of the monitored metric (e.g. validation loss
            or validation accuracy, depending on ``mode``).

        Examples
        --------
        >>> scheduler.step(val_loss)
        """
        if self.mode == "min":
            improved = metrics < self._best - self.threshold
        else:
            improved = metrics > self._best + self.threshold

        if improved:
            self._best = metrics
            self._num_bad_epochs = 0
        else:
            self._num_bad_epochs += 1
            if self._num_bad_epochs > self.patience:
                for group in self.optimizer.param_groups:
                    new_lr = max(group["lr"] * self.factor, self.min_lr)
                    group["lr"] = new_lr
                self.optimizer._sync_hyperparams()
                self._num_bad_epochs = 0


class NoamScheduler(_LRScheduler):
    r"""Noam learning rate schedule from the original Transformer paper.

    The learning rate increases linearly during a warmup phase and then
    decays proportionally to the inverse square root of the step number:

    .. math::

        \eta_t = d_{\text{model}}^{-0.5} \cdot
                  \min\!\left(t^{-0.5},\;
                                 t \cdot w^{-1.5}\right)

    where :math:`d_{\text{model}}` is the model dimensionality and
    :math:`w` is the number of warmup steps.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.  The ``lr`` in each param group is set to the
        Noam value directly (the base learning rate is not used as a
        multiplicative factor).
    d_model : int
        Dimensionality of the model (e.g. 512 for the base Transformer).
        Larger models have smaller peak learning rates.
    warmup_steps : int
        Number of warmup steps during which the LR increases linearly.
        A typical value is 4000.
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    d_model : int
        Model dimension used in the scaling formula.
    warmup_steps : int
        Warmup period length.

    Notes
    -----
    The Noam schedule is designed to be called **once per training step**
    (i.e. per batch), not once per epoch.  The learning rate peaks at step
    :math:`t^* = w` and then decreases as :math:`t^{-0.5}`.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98))
    >>> scheduler = optim.NoamScheduler(optimizer, d_model=512, warmup_steps=4000)
    >>> for step, batch in enumerate(dataloader):
    ...     train_step(batch)
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the NoamScheduler.  See the class docstring for parameter semantics."""
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        step = max(1, self.last_epoch)
        scale = self.d_model ** (-0.5) * min(
            step ** (-0.5), step * self.warmup_steps ** (-1.5)
        )
        return [scale] * len(self.optimizer.param_groups)


class MultiplicativeLR(_LRScheduler):
    r"""Multiply the learning rate by a factor returned by a function each epoch.

    Unlike :class:`LambdaLR`, which computes the learning rate as
    ``base_lr * fn(t)``, ``MultiplicativeLR`` **accumulates** the factor:
    each epoch the current learning rate is multiplied by ``lr_lambda(t)``:

    .. math::

        \eta_t = \eta_{t-1} \cdot \lambda(t)

    This means the schedule depends on the entire history of ``lr_lambda``
    values, not only the current epoch index.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    lr_lambda : callable
        A function ``f(epoch: int) -> float`` returning the multiplicative
        factor to apply at that epoch.
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    lr_lambda : callable
        The factor function applied each epoch.

    Notes
    -----
    At epoch 0 the learning rate is left unchanged (the factor is not
    applied on the very first step so that the initial LR from the
    optimizer is honored).

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    >>> # Decay by 0.95 every epoch
    >>> scheduler = optim.MultiplicativeLR(optimizer, lr_lambda=lambda t: 0.95)
    >>> for epoch in range(50):
    ...     train(...)
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable[[int], float],
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the MultiplicativeLR.  See the class docstring for parameter semantics."""
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        if self.last_epoch == 0:
            return [float(g["lr"]) for g in self.optimizer.param_groups]  # type: ignore[arg-type]  # lr is float at runtime
        factor = self.lr_lambda(self.last_epoch)
        return [g["lr"] * factor for g in self.optimizer.param_groups]


class LinearLR(_LRScheduler):
    r"""Linearly interpolate the learning rate from a start factor to an end factor.

    The learning rate is scaled by a factor that changes linearly from
    ``start_factor`` to ``end_factor`` over ``total_iters`` steps:

    .. math::

        \text{factor}(t) = \text{start\_factor}
            + \frac{(\text{end\_factor} - \text{start\_factor})
                     \cdot \min(t,\, T)}{T}

    .. math::

        \eta_t = \eta_0 \cdot \text{factor}(t)

    where :math:`T = \text{total\_iters}`.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    start_factor : float, optional
        The multiplier applied to the base LR at epoch 0
        (default: ``1/3``).
    end_factor : float, optional
        The multiplier applied to the base LR at epoch ``total_iters``
        (default: ``1.0``).
    total_iters : int, optional
        Number of steps over which the interpolation runs (default: ``5``).
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    start_factor : float
        Initial LR multiplier.
    end_factor : float
        Final LR multiplier.
    total_iters : int
        Interpolation length in epochs.

    Notes
    -----
    ``LinearLR`` is commonly used for **linear warmup**: set
    ``start_factor`` to a small value (e.g. ``1/total_iters``) and
    ``end_factor=1.0`` so the learning rate ramps up to its nominal value
    over the first few epochs.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    >>> # Warm up over 5 epochs: LR goes from lr/3 to lr
    >>> scheduler = optim.LinearLR(
    ...     optimizer, start_factor=1/3, end_factor=1.0, total_iters=5
    ... )
    >>> for epoch in range(50):
    ...     train(...)
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the LinearLR.  See the class docstring for parameter semantics."""
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        t = min(self.last_epoch, self.total_iters)
        factor = (
            self.start_factor
            + (self.end_factor - self.start_factor) * t / self.total_iters
        )
        return [base_lr * factor for base_lr in self.base_lrs]


class ConstantLR(_LRScheduler):
    r"""Hold the learning rate at a constant scaled value, then restore the base LR.

    For the first ``total_iters`` epochs the learning rate is scaled by
    ``factor``; after that it is restored to the original base learning rate:

    .. math::

        \eta_t =
        \begin{cases}
            \eta_0 \cdot \text{factor} & t < T \\
            \eta_0                        & t \ge T
        \end{cases}

    where :math:`T = \text{total\_iters}`.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    factor : float, optional
        Multiplicative factor applied to the base LR during the constant
        phase (default: ``1/3``).
    total_iters : int, optional
        Number of epochs to hold the scaled LR before restoring the base
        value (default: ``5``).
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    factor : float
        LR scaling factor during the constant phase.
    total_iters : int
        Duration of the constant phase in epochs.

    Notes
    -----
    ``ConstantLR`` is useful as the first stage in a :class:`SequentialLR`
    chain, e.g. to hold a reduced LR during an initial burn-in period
    before switching to a main decay schedule.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    >>> # Hold LR at 0.1/3 for 5 epochs, then jump back to 0.1
    >>> scheduler = optim.ConstantLR(optimizer, factor=1/3, total_iters=5)
    >>> for epoch in range(50):
    ...     train(...)
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = 1.0 / 3,
        total_iters: int = 5,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the ConstantLR.  See the class docstring for parameter semantics."""
        self.factor = factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        if self.last_epoch < self.total_iters:
            return [base_lr * self.factor for base_lr in self.base_lrs]
        return list(self.base_lrs)


class PolynomialLR(_LRScheduler):
    r"""Decay the learning rate using a polynomial function over a fixed number of steps.

    The learning rate follows a polynomial decay from the base LR down to
    ``eta_min`` over ``total_iters`` epochs:

    .. math::

        \eta_t =
        \begin{cases}
            (\eta_0 - \eta_{\min})
            \left(1 - \dfrac{t}{T}\right)^{p}
            + \eta_{\min}
            & t < T \\
            \eta_{\min} & t \ge T
        \end{cases}

    where :math:`T = \text{total\_iters}` and :math:`p = \text{power}`.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    total_iters : int, optional
        Number of epochs over which the decay runs (default: ``5``).
    power : float, optional
        Exponent of the polynomial.  ``power=1`` gives linear decay;
        ``power=2`` gives quadratic decay (default: ``1.0``).
    eta_min : float, optional
        Minimum learning rate reached at ``total_iters`` (default: ``0.0``).
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    total_iters : int
        Decay period in epochs.
    power : float
        Polynomial exponent.
    eta_min : float
        Lower bound on the learning rate.

    Notes
    -----
    ``power=1`` (linear) is equivalent to :class:`LinearLR` with
    ``start_factor=1`` and ``end_factor=0``.  Higher powers produce a
    schedule that decays slowly at first and more steeply near the end.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    >>> scheduler = optim.PolynomialLR(
    ...     optimizer, total_iters=100, power=2.0, eta_min=1e-5
    ... )
    >>> for epoch in range(100):
    ...     train(...)
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_iters: int = 5,
        power: float = 1.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the PolynomialLR.  See the class docstring for parameter semantics."""
        self.total_iters = total_iters
        self.power = power
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        if self.last_epoch >= self.total_iters:
            return [self.eta_min] * len(self.base_lrs)
        t = self.last_epoch
        T = self.total_iters
        return [
            (base_lr - self.eta_min) * ((1 - t / T) ** self.power) + self.eta_min
            for base_lr in self.base_lrs
        ]


class CosineAnnealingWarmRestarts(_LRScheduler):
    r"""Cosine annealing with periodic warm restarts (SGDR).

    Implements the Stochastic Gradient Descent with Warm Restarts (SGDR)
    schedule.  Within each restart cycle of length :math:`T_i` epochs the
    learning rate follows a cosine curve from the base LR down to
    ``eta_min``, then restarts:

    .. math::

        \eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})
                  \left(1 + \cos\!\left(
                      \frac{\pi\, T_{\text{cur}}}{T_i}
                  \right)\right)

    where :math:`T_{\text{cur}}` is the step count within the current
    cycle and :math:`T_i` is the current cycle length.  After each full
    cycle the cycle length is multiplied by ``T_mult``:

    .. math::

        T_{i+1} = T_i \cdot T_{\text{mult}}

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    T_0 : int
        Length (in epochs) of the first restart cycle.
    T_mult : int, optional
        Factor by which the cycle length is multiplied after each restart
        (default: ``1``, i.e. all cycles have the same length).
    eta_min : float, optional
        Minimum learning rate at the bottom of each cosine curve
        (default: ``0.0``).
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    T_0 : int
        Initial cycle length.
    T_mult : int
        Cycle-length multiplier applied after each restart.
    eta_min : float
        Lower bound on the learning rate.

    Notes
    -----
    With ``T_mult=1`` every cycle has the same length ``T_0``.  With
    ``T_mult=2`` cycle lengths double after each restart: ``T_0``,
    ``2*T_0``, ``4*T_0``, …  Longer later cycles are useful because the
    model can refine a good basin found in earlier cycles.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    >>> scheduler = optim.CosineAnnealingWarmRestarts(
    ...     optimizer, T_0=10, T_mult=2, eta_min=1e-5
    ... )
    >>> for epoch in range(80):
    ...     train(...)
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the CosineAnnealingWarmRestarts.  See the class docstring for parameter semantics."""
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self._T_cur = 0
        self._T_i = T_0
        super().__init__(optimizer, last_epoch, verbose)

    def step(self) -> None:
        """Advance the scheduler by one step and update the optimizer learning rates.

        Notes
        -----
        Should be called after the optimizer's ``.step()`` at the end of each
        epoch (or each iteration, depending on the schedule).
        """
        self.last_epoch += 1
        self._step_count += 1
        self._T_cur += 1
        if self._T_cur >= self._T_i:
            self._T_cur -= self._T_i
            self._T_i *= self.T_mult
        values = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, values):
            group["lr"] = lr
        self._last_lr = list(values)
        self.optimizer._sync_hyperparams()
        if self.verbose:
            self.print_lr(self.verbose, self.last_epoch, values)

    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self._T_cur / self._T_i))
            / 2
            for base_lr in self.base_lrs
        ]


class OneCycleLR(_LRScheduler):
    r"""1Cycle learning rate policy.

    Implements the 1cycle policy: the learning rate first rises from an
    initial value to ``max_lr`` over a warmup phase, then anneals down to
    a minimum value over the remaining steps.

    Three special learning rates are derived from ``max_lr``:

    .. math::

        \eta_{\text{init}} &= \frac{\eta_{\max}}{\text{div\_factor}} \\
        \eta_{\min}  &= \frac{\eta_{\text{init}}}{\text{final\_div\_factor}}

    The schedule has two phases:

    1. **Warmup** (first ``pct_start * total_steps`` steps): anneal from
       :math:`\eta_{\text{init}}` up to :math:`\eta_{\max}`.
    2. **Cooldown** (remaining steps): anneal from :math:`\eta_{\max}` down
       to :math:`\eta_{\min}`.

    Each phase uses either cosine or linear annealing depending on
    ``anneal_strategy``.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    max_lr : float
        Peak learning rate reached at the end of the warmup phase.
    total_steps : int
        Total number of steps (batches) in the training run.
    pct_start : float, optional
        Fraction of ``total_steps`` devoted to the warmup phase
        (default: ``0.3``).
    anneal_strategy : str, optional
        Annealing function: ``"cos"`` for cosine annealing or
        ``"linear"`` for linear annealing (default: ``"cos"``).
    div_factor : float, optional
        Determines the initial LR as ``max_lr / div_factor``
        (default: ``25.0``).
    final_div_factor : float, optional
        Determines the minimum LR as ``initial_lr / final_div_factor``
        (default: ``1e4``).
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    max_lr : float
        Peak learning rate.
    total_steps : int
        Total training steps.
    pct_start : float
        Warmup fraction.
    anneal_strategy : str
        ``"cos"`` or ``"linear"``.
    div_factor : float
        Initial LR divisor.
    final_div_factor : float
        Minimum LR divisor relative to initial LR.

    Notes
    -----
    The 1cycle policy should be called **once per training step** (per
    batch), not once per epoch.  It is designed for super-convergence
    and often allows training with much larger learning rates than
    standard schedules.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    >>> scheduler = optim.OneCycleLR(
    ...     optimizer, max_lr=0.1, total_steps=len(dataloader) * epochs
    ... )
    >>> for batch in dataloader:
    ...     train_step(batch)
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the OneCycleLR.  See the class docstring for parameter semantics."""
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        super().__init__(optimizer, last_epoch, verbose)

    def _annealing_cos(self, start: float, end: float, pct: float) -> float:
        return end + (start - end) / 2 * (math.cos(math.pi * pct) + 1)

    def _annealing_linear(self, start: float, end: float, pct: float) -> float:
        return start + pct * (end - start)

    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        anneal = (
            self._annealing_cos
            if self.anneal_strategy == "cos"
            else self._annealing_linear
        )
        t = self.last_epoch
        T = self.total_steps
        warmup_end = T * self.pct_start - 1.0  # end step of warmup phase (inclusive)
        cooldown_end = float(T - 1)
        init_lr = self.max_lr / self.div_factor
        min_lr = init_lr / self.final_div_factor
        results = []
        for _ in self.base_lrs:
            if t <= warmup_end:
                pct = t / max(1.0, warmup_end)
                lr = anneal(init_lr, self.max_lr, pct)
            else:
                pct = (t - warmup_end) / max(1.0, cooldown_end - warmup_end)
                lr = anneal(self.max_lr, min_lr, pct)
            results.append(lr)
        return results


class SequentialLR:
    """Chain multiple schedulers and switch between them at epoch milestones.

    ``SequentialLR`` runs its list of schedulers one at a time, activating
    the next scheduler whenever the epoch counter crosses a milestone.
    Before the first milestone, ``schedulers[0]`` is active; between
    milestone ``k-1`` and ``k``, ``schedulers[k]`` is active.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer shared by all schedulers.
    schedulers : list of _LRScheduler
        Ordered list of scheduler instances to activate in sequence.
        Must have ``len(schedulers) == len(milestones) + 1``.
    milestones : list of int
        Sorted list of epoch indices at which to switch to the next
        scheduler.
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).

    Attributes
    ----------
    optimizer : Optimizer
        The shared optimizer.
    schedulers : list of _LRScheduler
        All schedulers in activation order.
    milestones : list of int
        Epoch indices marking scheduler transitions.
    last_epoch : int
        Number of :meth:`step` calls completed.

    Notes
    -----
    A common pattern is to combine a short warmup with a long cosine decay:

    .. code-block:: python

        warmup   = optim.LinearLR(optimizer, start_factor=0.1, total_iters=5)
        cosine   = optim.CosineAnnealingLR(optimizer, T_max=95)
        combined = optim.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[5],
        )

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    >>> s1 = optim.ConstantLR(optimizer, factor=0.1, total_iters=5)
    >>> s2 = optim.ExponentialLR(optimizer, gamma=0.9)
    >>> scheduler = optim.SequentialLR(optimizer, schedulers=[s1, s2], milestones=[5])
    >>> for epoch in range(50):
    ...     train(...)
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: list[_LRScheduler],
        milestones: list[int],
        last_epoch: int = -1,
    ) -> None:
        """Initialise the SequentialLR.  See the class docstring for parameter semantics."""
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.milestones = milestones
        self.last_epoch = last_epoch
        self._idx = 0

    def step(self) -> None:
        """Advance the active scheduler by one epoch, switching if a milestone is reached.

        Increments the internal epoch counter, checks whether any
        milestone has been crossed, and delegates to the currently active
        scheduler's :meth:`~_LRScheduler.step`.

        Examples
        --------
        >>> for epoch in range(100):
        ...     optimizer.step()
        ...     scheduler.step()
        """
        self.last_epoch += 1
        while (
            self._idx < len(self.milestones)
            and self.last_epoch >= self.milestones[self._idx]
        ):
            self._idx += 1
        self.schedulers[min(self._idx, len(self.schedulers) - 1)].step()

    def get_last_lr(self) -> list[float]:
        """Return the last learning rates from the currently active scheduler.

        Returns
        -------
        list of float
            Current learning rate of each optimizer param group, as
            reported by the active scheduler.
        """
        return self.schedulers[min(self._idx, len(self.schedulers) - 1)].get_last_lr()


class ChainedScheduler:
    r"""Apply multiple schedulers simultaneously, compounding their LR changes.

    At every :meth:`step` call each scheduler in the list is stepped in
    order.  Because each scheduler writes its own learning rates back into
    the shared optimizer, the net effect is the **composition** of all
    individual schedule factors:

    .. math::

        \eta_t = \eta_0 \cdot \prod_{k} f_k(t)

    where :math:`f_k(t)` is the multiplicative change applied by scheduler
    :math:`k` at epoch :math:`t`.

    Parameters
    ----------
    schedulers : list of _LRScheduler
        List of scheduler instances to step together.  All schedulers must
        wrap the same optimizer.

    Attributes
    ----------
    schedulers : list of _LRScheduler
        The chained schedulers stepped in order.

    Notes
    -----
    ``ChainedScheduler`` differs from :class:`SequentialLR` in that **all**
    schedulers are active at every step rather than one at a time.  This is
    useful when you want, for example, a warmup schedule and an exponential
    decay to both apply simultaneously.

    Only the last scheduler in the chain is consulted by
    :meth:`get_last_lr`, so the returned value reflects that scheduler's
    view of the current LR (which is also what is stored in the optimizer).

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    >>> warmup = optim.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    >>> decay  = optim.ExponentialLR(optimizer, gamma=0.95)
    >>> scheduler = optim.ChainedScheduler([warmup, decay])
    >>> for epoch in range(50):
    ...     train(...)
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(self, schedulers: list[_LRScheduler]) -> None:
        """Initialise the ChainedScheduler.  See the class docstring for parameter semantics."""
        self.schedulers = schedulers

    def step(self) -> None:
        """Step all chained schedulers simultaneously.

        Calls :meth:`~_LRScheduler.step` on every scheduler in order.
        Each scheduler writes its updated learning rates into the shared
        optimizer, so the final LR reflects the composition of all
        schedules.

        Examples
        --------
        >>> for epoch in range(50):
        ...     optimizer.step()
        ...     scheduler.step()
        """
        for sched in self.schedulers:
            sched.step()

    def get_last_lr(self) -> list[float]:
        """Return the last learning rates from the final scheduler in the chain.

        Returns
        -------
        list of float
            Current learning rate of each optimizer param group, as
            reported by the last scheduler in :attr:`schedulers`.
        """
        return self.schedulers[-1].get_last_lr()
