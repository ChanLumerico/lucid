"""
Flatten and Unflatten modules.
"""

from typing import Any
from lucid.nn.module import Module
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap


class Flatten(Module):
    """Flatten consecutive dimensions into one."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Any) -> Any:
        return _wrap(_C_engine.flatten(_unwrap(x), self.start_dim, self.end_dim))

    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"


class Unflatten(Module):
    """Unflatten one dimension into multiple dimensions."""

    def __init__(self, dim: int, unflattened_size: tuple[int, ...]) -> None:
        super().__init__()
        self.dim = dim
        self.unflattened_size = unflattened_size

    def forward(self, x: Any) -> Any:
        shape = list(x.shape)
        new_shape = shape[: self.dim] + list(self.unflattened_size) + shape[self.dim + 1:]
        return _wrap(_C_engine.reshape(_unwrap(x), new_shape))

    def extra_repr(self) -> str:
        return f"dim={self.dim}, unflattened_size={self.unflattened_size}"
