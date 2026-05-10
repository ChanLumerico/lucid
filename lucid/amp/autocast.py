"""
autocast context manager for automatic mixed precision.
"""

import functools
from typing import Callable, TypeVar
from lucid._C import engine as _C_engine
from lucid._dtype import dtype, float16, to_engine_dtype

_F = TypeVar("_F", bound=Callable[..., object])


class autocast:
    """Enable automatic mixed-precision computation.

    Operations that support it will cast inputs to the specified dtype,
    reducing memory and potentially increasing throughput on Metal GPU.

    The C++ AutocastGuard.__exit__ does not restore state, so this class
    implements proper RAII entirely in Python.

    Args:
        device_type: Device context ('metal' or 'cpu'). Default: 'metal'.
        dtype:       Target dtype for autocast (default: float16).
        enabled:     If False, this context manager is a no-op.

    Example:
        with lucid.amp.autocast():
            output = model(input)
    """

    def __init__(
        self,
        device_type: str = "metal",
        dtype: dtype = float16,
        enabled: bool = True,
    ) -> None:
        self._dtype = dtype
        self._enabled = enabled
        self._prev_active: bool = False
        self._prev_dtype: object = None

    def __enter__(self) -> autocast:
        if not self._enabled:
            return self
        self._prev_active = _C_engine.amp_is_active()
        self._prev_dtype = _C_engine.amp_active_dtype()
        engine_dtype = to_engine_dtype(self._dtype)
        self._guard = _C_engine.AutocastGuard(engine_dtype)
        self._guard.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        if not self._enabled:
            return
        if self._prev_active and self._prev_dtype is not None:
            # Restore the previous AMP dtype (e.g. nested autocast blocks).
            prev_guard = _C_engine.AutocastGuard(self._prev_dtype)  # type: ignore[arg-type]
            prev_guard.__enter__()
        elif not self._prev_active:
            # AMP was off before this context.  The engine has no disable_amp()
            # API, so restore neutrality by entering a float32 guard — ops will
            # cast to float32 (identity for most) until the guard is superseded
            # or the program exits AMP-active scope.
            restore_guard = _C_engine.AutocastGuard(_C_engine.F32)
            restore_guard.__enter__()

    def __call__(self, fn: _F) -> _F:
        """Use as a function decorator."""

        @functools.wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> object:
            with autocast(dtype=self._dtype, enabled=self._enabled):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    @staticmethod
    def is_autocast_enabled() -> bool:
        """Return True if autocast is currently active."""
        return _C_engine.amp_is_active()

    @staticmethod
    def get_autocast_dtype() -> dtype | None:
        """Return the currently active autocast dtype, or None."""
        from lucid._dtype import _ENGINE_TO_DTYPE

        eng = _C_engine.amp_active_dtype()
        if eng is None:
            return None
        return _ENGINE_TO_DTYPE.get(eng)
