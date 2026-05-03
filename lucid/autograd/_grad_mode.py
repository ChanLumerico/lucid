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

    Can be used as a context manager or as a function decorator.

    Examples:
        with lucid.no_grad():
            y = model(x)

        @lucid.no_grad()
        def eval_step(x):
            return model(x)
    """

    def __enter__(self) -> "no_grad":
        self._guard = _C_engine.NoGradGuard()
        self._guard.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self._guard.__exit__(*args)

    def __call__(self, fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with no_grad():
                return fn(*args, **kwargs)
        return wrapper  # type: ignore[return-value]


class enable_grad:
    """Re-enable gradient computation inside a no_grad block."""

    def __enter__(self) -> "enable_grad":
        _C_engine.set_grad_enabled(True)
        return self

    def __exit__(self, *args: Any) -> None:
        pass


def set_grad_enabled(flag: bool) -> None:
    """Globally enable or disable gradient computation."""
    _C_engine.set_grad_enabled(flag)


def is_grad_enabled() -> bool:
    """Return True if gradient computation is currently enabled."""
    return _C_engine.grad_enabled()


@contextmanager
def inference_mode() -> Iterator[None]:
    """Alias for no_grad; disables gradient tracking."""
    with no_grad():
        yield
