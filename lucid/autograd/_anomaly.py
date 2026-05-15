"""
detect_anomaly context manager for autograd debugging.

When active, NaN/Inf values in backward gradients will raise a RuntimeError
with a descriptive message.  This is primarily useful for debugging gradient
computation.

Note: Full anomaly detection (backward stack traces) requires engine-level
hooks that are not yet wired.  This implementation provides the Python
context-manager API so code that uses ``detect_anomaly`` does not break;
the NaN/Inf check is done on the Python side at the end of each backward call.
"""


class detect_anomaly:
    r"""Context manager / decorator that enables autograd anomaly detection.

    Reverse-mode automatic differentiation propagates a chain of partial
    derivatives backwards through the computation graph; a single
    ``NaN`` or ``Inf`` appearing anywhere in that chain silently
    contaminates every upstream gradient and is one of the hardest
    classes of training bugs to localise. ``detect_anomaly`` opts the
    current scope into stricter checking: after each call to
    :func:`backward` the resulting gradients are scanned for
    non-finite values, and an exception is raised at the offending
    boundary so the failure surfaces at the source rather than at the
    optimiser step.

    Enabling anomaly detection adds overhead — every backward pass
    performs an additional reduction over the gradient tensors — so it
    is intended for debugging sessions, not steady-state training.

    Parameters
    ----------
    check_nan : bool, optional
        If ``True`` (default) every backward pass is checked for
        ``NaN`` / ``Inf`` gradients and a ``RuntimeError`` is raised on
        the first violation. Setting it to ``False`` enters the context
        but performs no checking — useful for nested scopes where an
        outer block already enabled the flag.

    Attributes
    ----------
    check_nan : bool
        The mode argument as passed to ``__init__``.

    Notes
    -----
    Reverse-mode AD computes

    .. math::

        \frac{\partial \mathcal{L}}{\partial x}
        = \sum_{p \in \text{paths}(x \to \mathcal{L})}
            \prod_{(u, v) \in p} \frac{\partial v}{\partial u}.

    A single non-finite Jacobian entry along any path corrupts the
    sum, so the check is performed on the final accumulated gradient
    rather than on individual op outputs.

    The context manager restores the previous anomaly flag on exit,
    so nested ``with`` blocks behave correctly.

    Examples
    --------
    Use as a context manager during loss computation:

    >>> import lucid
    >>> from lucid.autograd import detect_anomaly
    >>> x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> with detect_anomaly():
    ...     y = (x * x).sum()
    ...     y.backward()

    Use as a decorator on a training step:

    >>> @detect_anomaly()
    ... def train_step(x, target):
    ...     loss = ((x - target) ** 2).sum()
    ...     loss.backward()
    ...     return loss
    """

    def __init__(self, check_nan: bool = True) -> None:
        """Initialise the instance.  See the class docstring for parameter semantics."""
        self.check_nan = check_nan
        self._prev: bool = False

    def __enter__(self) -> detect_anomaly:
        """Enter the context.  Returns self so the value can be bound via ``with ... as``."""
        from lucid.autograd._grad_mode import _ANOMALY_ENABLED

        self._prev = _ANOMALY_ENABLED[0]
        _ANOMALY_ENABLED[0] = self.check_nan
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the context, restoring any state that was modified on entry."""
        from lucid.autograd._grad_mode import _ANOMALY_ENABLED

        _ANOMALY_ENABLED[0] = self._prev

    def __call__(self, fn: object) -> object:
        """Support use as a decorator."""
        import functools

        @functools.wraps(fn)  # type: ignore[arg-type]
        def wrapper(*args: object, **kwargs: object) -> object:
            """Decorator-generated wrapper that applies the surrounding behaviour to the wrapped callable."""
            with self.__class__(self.check_nan):
                return fn(*args, **kwargs)

        return wrapper


def set_detect_anomaly(mode: bool, check_nan: bool = True) -> None:
    r"""Programmatic global toggle for autograd anomaly detection.

    Equivalent to entering :class:`detect_anomaly` once and never
    exiting — flips the process-wide flag that :func:`backward`
    consults at the end of each pass. Prefer the ``with``-block form
    when the scope is bounded; use this free function when the flag
    is controlled by a CLI argument or an environment variable read
    once at start-up.

    Parameters
    ----------
    mode : bool
        ``True`` enables ``NaN`` / ``Inf`` checking on every backward;
        ``False`` disables it.
    check_nan : bool, optional
        Forwarded to the underlying flag. Currently the only supported
        check is for non-finite values; the parameter exists for API
        symmetry and forward compatibility.

    Returns
    -------
    None
        The function mutates a module-level flag; it has no return
        value.

    Notes
    -----
    The effective flag is :math:`\text{mode} \land \text{check\_nan}`:
    setting either to ``False`` disables checking. There is no stack of
    saved states — once changed, the flag stays until the next call.

    Examples
    --------
    Enable checks at start-up:

    >>> import lucid
    >>> from lucid.autograd import set_detect_anomaly, is_anomaly_enabled
    >>> set_detect_anomaly(True)
    >>> is_anomaly_enabled()
    True
    >>> set_detect_anomaly(False)
    """
    from lucid.autograd._grad_mode import _ANOMALY_ENABLED

    _ANOMALY_ENABLED[0] = bool(mode) and bool(check_nan)


def is_anomaly_enabled() -> bool:
    r"""Return whether autograd anomaly detection is currently enabled.

    Query accessor for the global flag that :class:`detect_anomaly`
    and :func:`set_detect_anomaly` mutate. Useful when conditionally
    wrapping additional debug logic, or when asserting state inside
    test suites.

    Parameters
    ----------
    None

    Returns
    -------
    bool
        ``True`` if anomaly detection is currently active; ``False``
        otherwise.

    Notes
    -----
    The flag is process-global. In multi-threaded code (Lucid does
    not currently run multiple Python threads through the autograd
    engine) the value would be shared across threads — treat it as
    a debug switch rather than per-call state.

    Examples
    --------
    >>> import lucid
    >>> from lucid.autograd import detect_anomaly, is_anomaly_enabled
    >>> is_anomaly_enabled()
    False
    >>> with detect_anomaly():
    ...     is_anomaly_enabled()
    True
    >>> is_anomaly_enabled()
    False
    """
    from lucid.autograd._grad_mode import _ANOMALY_ENABLED

    return bool(_ANOMALY_ENABLED[0])
