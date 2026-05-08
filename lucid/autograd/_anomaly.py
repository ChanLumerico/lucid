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
    """Context manager / decorator that enables anomaly detection.

    When used as a context manager, any NaN or Inf values produced by
    ``backward()`` will raise a ``RuntimeError``.

    Example::

        with lucid.autograd.detect_anomaly():
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
    """

    def __init__(self, check_nan: bool = True) -> None:
        self.check_nan = check_nan
        self._prev: bool = False

    def __enter__(self) -> detect_anomaly:
        from lucid.autograd._grad_mode import _ANOMALY_ENABLED

        self._prev = _ANOMALY_ENABLED[0]
        _ANOMALY_ENABLED[0] = self.check_nan
        return self

    def __exit__(self, *args: object) -> None:
        from lucid.autograd._grad_mode import _ANOMALY_ENABLED

        _ANOMALY_ENABLED[0] = self._prev

    def __call__(self, fn: object) -> object:
        """Support use as a decorator."""
        import functools

        @functools.wraps(fn)  # type: ignore[arg-type]
        def wrapper(*args: object, **kwargs: object) -> object:
            with self.__class__(self.check_nan):
                return fn(*args, **kwargs)  # type: ignore[operator]

        return wrapper


def set_detect_anomaly(mode: bool, check_nan: bool = True) -> None:
    """Programmatic toggle for autograd anomaly detection.

    Equivalent to entering / exiting ``detect_anomaly()`` once, but without
    requiring a ``with`` block.  Useful for enabling anomaly checks at
    program start (e.g. behind a debug flag) and leaving them on for the
    remainder of the run.

    Parameters
    ----------
    mode : bool
        ``True`` enables NaN / Inf checking on every backward; ``False``
        disables it.
    check_nan : bool, optional
        Forwarded to the underlying flag.  Currently the only supported
        check is for NaN / Inf values; included for API symmetry with
        the reference framework.
    """
    from lucid.autograd._grad_mode import _ANOMALY_ENABLED

    _ANOMALY_ENABLED[0] = bool(mode) and bool(check_nan)


def is_anomaly_enabled() -> bool:
    """Return whether autograd anomaly detection is currently enabled."""
    from lucid.autograd._grad_mode import _ANOMALY_ENABLED

    return bool(_ANOMALY_ENABLED[0])
