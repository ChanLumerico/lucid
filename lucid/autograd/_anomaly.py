"""
detect_anomaly context manager — mirrors torch.autograd.detect_anomaly.

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

    def __call__(self, fn):
        """Support use as a decorator."""
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self.__class__(self.check_nan):
                return fn(*args, **kwargs)

        return wrapper
