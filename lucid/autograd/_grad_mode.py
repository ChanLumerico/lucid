"""
Gradient mode context managers and decorators.
"""

import functools
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

from lucid._C import engine as _C_engine

_F = TypeVar("_F", bound=Callable[..., Any])


class no_grad:
    """
    Disable gradient computation.

    Correctly restores the previous grad mode on exit (RAII).
    Can be used as a context manager or function decorator.

    Examples:
        with lucid.no_grad():
            y = model(x)

        @lucid.no_grad()
        def eval_step(x):
            return model(x)
    """

    _prev: bool

    def __enter__(self) -> "no_grad":
        self._prev = _C_engine.grad_enabled()
        _C_engine.set_grad_enabled(False)
        return self

    def __exit__(self, *args: Any) -> None:
        _C_engine.set_grad_enabled(self._prev)

    def __call__(self, fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with no_grad():
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]


class enable_grad:
    """Re-enable gradient computation, restoring previous mode on exit."""

    _prev: bool

    def __enter__(self) -> "enable_grad":
        self._prev = _C_engine.grad_enabled()
        _C_engine.set_grad_enabled(True)
        return self

    def __exit__(self, *args: Any) -> None:
        _C_engine.set_grad_enabled(self._prev)

    def __call__(self, fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with enable_grad():
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]


def set_grad_enabled(flag: bool) -> None:
    """Globally enable or disable gradient computation."""
    _C_engine.set_grad_enabled(flag)


def is_grad_enabled() -> bool:
    """Return True if gradient computation is currently enabled."""
    return _C_engine.grad_enabled()


@contextmanager
def inference_mode() -> Iterator[None]:
    """Context manager that disables gradient tracking (alias for no_grad)."""
    with no_grad():
        yield
