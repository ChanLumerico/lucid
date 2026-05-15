"""Thread-local vmap execution context.

Tracks the active ``randomness`` mode so that random op entry points can
raise immediately when ``randomness='error'`` is set by an enclosing
``vmap`` call.  The module has **zero** dependencies on the rest of Lucid
so it is safe to import from anywhere without circular-import risk.
"""

import threading
from types import TracebackType

_local: threading.local = threading.local()


def get_randomness() -> str:
    """Return the active vmap randomness mode, defaulting to ``'different'``."""
    return str(getattr(_local, "randomness", "different"))


def set_randomness(mode: str) -> None:
    """Set the active vmap randomness mode on the current thread."""
    _local.randomness = mode


class _RandomnessGuard:
    """RAII guard: push *mode* as the vmap randomness context, pop on exit.

    Nested vmap calls each push their own mode so the innermost vmap's
    ``randomness`` setting takes effect for random ops inside it.
    """

    def __init__(self, mode: str) -> None:
        self._mode = mode
        self._prev: str = "different"

    def __enter__(self) -> None:
        self._prev = get_randomness()
        set_randomness(self._mode)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        set_randomness(self._prev)


def check_random_allowed() -> None:
    """Raise ``RuntimeError`` if the current vmap context forbids random ops.

    Call this at the top of every Python-level random factory
    (``randn``, ``rand``, ``randint``, …) so that ``vmap(...,
    randomness='error')`` produces a clear error instead of silently
    sampling inside the vectorised function.
    """
    if get_randomness() == "error":
        raise RuntimeError(
            "vmap: called a random op inside a vmapped function with "
            "randomness='error'. Pass randomness='different' to allow "
            "independent random draws per batch element, or "
            "randomness='same' for a shared stream across elements."
        )
