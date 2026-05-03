"""
Removable hook handles for forward/backward hooks on Module.
"""

from typing import Callable


class RemovableHandle:
    """Handle returned by register_*_hook(); call .remove() to deregister."""

    def __init__(self, hooks: dict[int, Callable[..., object]], key: int) -> None:
        self._hooks = hooks
        self._key = key

    def remove(self) -> None:
        """Remove this hook from the module."""
        self._hooks.pop(self._key, None)

    def __enter__(self) -> "RemovableHandle":
        return self

    def __exit__(self, *args: object) -> None:
        self.remove()
