"""
The learning-rate scheduler base classes and the rules the schedulers share.
"""

from collections.abc import Sequence

from lucid.optim.optimizer import Optimizer, _group_float


def _holds_a_callable(value: object) -> bool:
    """Whether ``value`` carries a Python function into a checkpoint.

    ``not callable(v)`` was the whole test, and ``LambdaLR`` stores its
    lambdas as ``lr_lambdas`` — a *list* — which is not itself callable.
    So the one scheduler the exclusion was written for was the one it
    missed, and ``lucid.save`` on a checkpoint holding a ``LambdaLR``
    raised out of ``pickle``.  A container is disqualified by any
    callable in it, not by all of them: a list with one lambda in it is
    exactly as unpicklable as a list of nothing else.
    """
    if callable(value):
        return True
    if isinstance(value, (list, tuple)):
        return any(callable(item) for item in value)
    return False


class _CheckpointState:
    """A scheduler's checkpoint: its attributes, less the optimizer and callables."""

    def state_dict(self) -> dict[str, object]:
        """Serialisable scheduler state, for checkpoint / resume.

        Mirrors the optimizer's contract: everything needed to continue the
        schedule exactly, minus the ``optimizer`` itself (the caller already
        holds it) and minus any callable hyper-parameter.

        Callables — ``LambdaLR``'s and ``MultiplicativeLR``'s ``lr_lambda`` —
        are **not** captured: a Python function is not reliably serialisable,
        and silently pickling one would make a checkpoint that only reloads in
        the process that wrote it.  Reconstruct the scheduler with the same
        lambda, then :meth:`load_state_dict` the rest.  This holds whether the
        callables are stored singly (``MultiplicativeLR.lr_lambda``) or in a
        list (``LambdaLR.lr_lambdas``, one per param group).

        Returns
        -------
        dict
            Attribute snapshot; feed it back to :meth:`load_state_dict`.
        """
        return {
            k: v
            for k, v in self.__dict__.items()
            if k != "optimizer" and not _holds_a_callable(v)
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore state produced by :meth:`state_dict`.

        Parameters
        ----------
        state_dict : dict
            A snapshot from :meth:`state_dict` of a scheduler of the same type.

        Notes
        -----
        Without this the schedule silently restarted on resume while the
        optimizer state restored correctly — the two went out of sync.
        """
        self.__dict__.update(state_dict)


class _LRScheduler(_CheckpointState):
    """Base class for all learning rate schedulers.

    Subclasses must implement :meth:`get_lr`, which returns a list of new
    learning rates — one per optimizer parameter group — given the current
    ``last_epoch`` counter.

    The scheduler tracks ``base_lrs``: the initial learning rates captured
    from the optimizer at construction time.  Every call to :meth:`step`
    increments ``last_epoch`` by one, recomputes the learning rates via
    :meth:`get_lr`, and writes them back into the optimizer's param groups.

    Most schedules are *chainable*: each step scales every group's current
    rate, so two of them on one optimizer compose (:class:`ChainedScheduler`)
    and a group added later with ``add_param_group`` follows the schedule
    from the rate it was added with.  The schedules that hold something per
    group from construction — a base rate, a lambda, a cycle's bounds:
    :class:`LambdaLR`, :class:`MultiplicativeLR`, :class:`CyclicLR`,
    :class:`OneCycleLR`, :class:`CosineAnnealingWarmRestarts` — have nothing
    for such a group and refuse the step with ``ValueError``.

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

    #: ``True`` only while the constructor writes step 0's rates.  A
    #: chainable schedule then leaves the rates as they are — a second
    #: scheduler on the optimizer must not apply its own step 0 on top of the
    #: first's, and a resumed one (``last_epoch >= 0``) not a step twice.
    _is_initial: bool = False

    def __init__(
        self, optimizer: Optimizer, last_epoch: int = -1, verbose: bool = False
    ) -> None:
        self.optimizer = optimizer
        # After N step() calls, last_epoch == N (the constructor's initial
        # step takes it from -1 to 0).
        self.last_epoch = last_epoch
        self.verbose = verbose
        # The rate before any scheduler touched it, kept on the group: the
        # schedule below writes step 0's rate straight away, and a second
        # scheduler on the same optimizer — a SequentialLR's cosine after
        # its warmup — must still start from the true base, not from that.
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        self.base_lrs: list[float] = [
            _group_float(g, "initial_lr" if "initial_lr" in g else "lr")
            for g in optimizer.param_groups
        ]
        self._last_lr: list[float] = []
        # Step 0's rate takes effect now, before the first optimizer step.
        # Only step() used to write a rate, so the first update ran at the
        # base rate whatever the schedule said: LambdaLR and LinearLR warmups
        # at full rate, ConstantLR unscaled, CyclicLR at max_lr, OneCycleLR at
        # max_lr rather than max_lr / div_factor — 25x its first step.
        self._initial_step()

    def _initial_step(self) -> None:
        """Write step 0's rates, in force before the first ``step()``."""
        self._step_count = 0
        self._is_initial = True
        try:
            self._update_lr()
        finally:
            self._is_initial = False

    def _group_lrs(self) -> list[float]:
        """Each param group's current rate — what a chainable step scales."""
        return [_group_float(g) for g in self.optimizer.param_groups]

    def _unrated_group(self, rated: int) -> ValueError:
        """The error for a step that has rates for only the first ``rated`` groups.

        A schedule that computes from what it captured at construction —
        ``base_lrs``, one lambda or one bound per group — has nothing for a
        group added afterwards.  The rates used to be written by a
        non-strict ``zip``, which left that group at the rate it was added
        with: unscheduled, and silently so.  The reference refuses the step.
        """
        return ValueError(
            f"{type(self).__name__}: param group {rated} was added after the "
            f"scheduler was built, and this schedule rates only the {rated} "
            "group(s) it was built over; build the scheduler after "
            "add_param_group"
        )

    def _update_lr(self, epoch: int | None = None) -> None:
        """Move to the next epoch — or to ``epoch``, by the closed form — and write it.

        Every group gets a rate or none does: a refused step leaves the
        rates and ``last_epoch`` as they were.
        """
        previous = self.last_epoch
        self.last_epoch = previous + 1 if epoch is None else epoch
        try:
            values = self.get_lr() if epoch is None else self._get_closed_form_lr()
            if len(values) < len(self.optimizer.param_groups):
                raise self._unrated_group(len(values))
        except ValueError:
            self.last_epoch = previous
            raise
        for group, lr in zip(self.optimizer.param_groups, values, strict=True):
            group["lr"] = lr
        self._last_lr = list(values)
        self.optimizer._sync_hyperparams()

    def _get_closed_form_lr(self) -> list[float]:
        """The rates at ``last_epoch`` from ``base_lrs`` alone, for a jump to it.

        :class:`SequentialLR` starts each child after the first at epoch 0
        this way.  A schedule with no closed form takes the step instead.
        """
        return self.get_lr()

    def _restart(self) -> None:
        """Start over at epoch 0, from the base rates — a SequentialLR milestone."""
        self._update_lr(0)

    def _undo_initial_step(self) -> None:
        """Take back the step 0 the constructor took, as SequentialLR does."""
        self.last_epoch -= 1

    def step(self) -> None:
        """Advance the scheduler by one epoch and update optimizer learning rates.

        Increments ``last_epoch`` by one, calls :meth:`get_lr` to obtain
        the new per-group learning rates, and writes them into the optimizer's
        param groups.  Also calls ``optimizer._sync_hyperparams()`` so that
        any engine-level optimizer mirrors the updated value.

        Should be called **once per epoch**, after ``optimizer.step()``.

        Examples
        --------
        >>> import lucid
        >>> import lucid.nn as nn
        >>> import lucid.optim as optim
        >>> model = nn.Linear(4, 2)
        >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
        >>> scheduler = lucid.optim.StepLR(optimizer, step_size=10)
        >>> for epoch in range(50):
        ...     optimizer.step()
        ...     scheduler.step()
        """
        self._update_lr()
        self._step_count += 1
        if self.verbose:
            self.print_lr(self.verbose, self.last_epoch, self._last_lr)

    def get_lr(self) -> list[float]:
        """Compute new learning rates for all param groups.

        Returns
        -------
        list of float
            One learning rate per optimizer param group, from each group's
            current rate (a chainable schedule) or from ``base_lrs`` and the
            current ``last_epoch`` (a closed-form one).

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
        """
        return [_group_float(g) for g in self.optimizer.param_groups]

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


class _CompositeSchedulerMixin:
    """What the composite schedulers share: checkpoints and taking back step 0."""

    optimizer: Optimizer
    schedulers: Sequence[_LRScheduler | _CompositeSchedulerMixin]

    def state_dict(self) -> dict[str, object]:
        """Serialisable state, including each wrapped scheduler's own state."""
        own = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("optimizer", "schedulers") and not _holds_a_callable(v)
        }
        own["schedulers"] = [sch.state_dict() for sch in self.schedulers]
        return own

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore state produced by :meth:`state_dict`."""
        incoming = dict(state_dict)
        raw_children = incoming.pop("schedulers", None)
        self.__dict__.update(incoming)
        if raw_children is not None:
            if not isinstance(raw_children, list):
                raise TypeError("load_state_dict: 'schedulers' must be a list")
            child_states: list[dict[str, object]] = raw_children
            if len(child_states) != len(self.schedulers):
                raise ValueError(
                    f"{type(self).__name__}.load_state_dict: got {len(child_states)} child states "
                    f"for {len(self.schedulers)} schedulers"
                )
            for sch, child in zip(self.schedulers, child_states, strict=True):
                sch.load_state_dict(child)

    def _undo_initial_step(self) -> None:
        """Take back each wrapped scheduler's step 0, as the reference recurses."""
        for sched in self.schedulers:
            sched._undo_initial_step()


def _check_schedulers(
    owner: str,
    schedulers: Sequence[_LRScheduler | _CompositeSchedulerMixin],
    optimizer: Optimizer | None,
    accepted: tuple[type, ...],
) -> Optimizer:
    """The optimizer a composite's children share, or why it cannot wrap them.

    Checked before the composite changes anything.  SequentialLR used to
    put every rate back to its base and take back its children's step 0
    first, then fail on a child it could not drive with an untyped
    ``AttributeError`` — leaving the optimizer and the children altered.
    A scheduler that steps on a metric is a scheduler, just not one a
    composite can step: ``ValueError``, as the reference raises.
    """
    if not schedulers:
        raise ValueError(f"{owner} expects at least one scheduler")
    for i, sched in enumerate(schedulers):
        if getattr(sched, "_steps_on_a_metric", False):
            raise ValueError(
                f"{owner} at index {i}: {type(sched).__name__} steps on a metric, "
                f"which {owner}.step() does not take"
            )
        if not isinstance(sched, accepted):
            kinds = {
                _LRScheduler: "an LR scheduler",
                _CompositeSchedulerMixin: "SequentialLR or ChainedScheduler",
            }
            names = " or ".join(kinds.get(t, t.__name__) for t in accepted)
            raise TypeError(
                f"{owner} at index {i}: expected {names}, got {type(sched).__name__}"
            )
    shared = optimizer or schedulers[0].optimizer
    for i, sched in enumerate(schedulers):
        if sched.optimizer is not shared:
            raise ValueError(
                f"{owner} expects all schedulers to belong to the same optimizer; "
                f"the one at index {i} does not"
            )
    return shared


def _require_group_keys(
    owner: str, optimizer: Optimizer, keys: tuple[str, ...]
) -> None:
    """Refuse a step whose schedule reads ``keys`` off a group that lacks them.

    The schedule sets them on the groups it was built over, so a group added
    afterwards has none unless it brought its own.
    """
    for i, group in enumerate(optimizer.param_groups):
        missing = [k for k in keys if k not in group]
        if missing:
            raise ValueError(
                f"{owner}: param group {i} has no {', '.join(missing)}, which the "
                "schedule reads off each group and sets only on the groups it was "
                "built over; build the scheduler after add_param_group, or give "
                f"the group {', '.join(keys)}"
            )


def _reset_to_base(optimizer: Optimizer) -> None:
    """Put each group back at the rate it had before any scheduler touched it."""
    for group in optimizer.param_groups:
        group["lr"] = _group_float(
            group, "initial_lr" if "initial_lr" in group else "lr"
        )
