"""
Padding modules: Constant, Reflection, Replication, Zero padding.
"""

from lucid._tensor.tensor import Tensor
from lucid._types import _Size2d
from lucid.nn.module import Module
from lucid.nn.functional.sampling import pad


class _ConstantPadNd(Module):
    """Base class for constant padding modules."""

    def __init__(self, padding: int | tuple[int, ...], value: float) -> None:
        super().__init__()
        self.padding = (
            padding if isinstance(padding, tuple) else _make_tuple(padding, self._dims)
        )
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        return pad(x, self.padding, mode="constant", value=self.value)

    def extra_repr(self) -> str:
        return f"padding={self.padding}, value={self.value}"


def _make_tuple(v: int, n: int) -> tuple[int, ...]:
    return (v,) * n


class ConstantPad1d(_ConstantPadNd):
    """Pad a 3-D tensor with a constant value on left/right."""

    _dims = 2

    def __init__(self, padding: _Size2d, value: float) -> None:
        super().__init__(padding, value)


class ConstantPad2d(_ConstantPadNd):
    """Pad a 4-D tensor with a constant value (left, right, top, bottom)."""

    _dims = 4

    def __init__(self, padding: int | tuple[int, int, int, int], value: float) -> None:
        super().__init__(padding, value)


class ConstantPad3d(_ConstantPadNd):
    """Pad a 5-D tensor with a constant value."""

    _dims = 6

    def __init__(
        self, padding: int | tuple[int, int, int, int, int, int], value: float
    ) -> None:
        super().__init__(padding, value)


class ZeroPad2d(ConstantPad2d):
    """Pad a 4-D tensor with zeros."""

    def __init__(self, padding: int | tuple[int, int, int, int]) -> None:
        super().__init__(padding, value=0.0)

    def extra_repr(self) -> str:
        return f"padding={self.padding}"


class ReflectionPad1d(Module):
    """Pad a 3-D tensor using reflection."""

    def __init__(self, padding: _Size2d) -> None:
        super().__init__()
        self.padding = (
            (padding, padding) if isinstance(padding, int) else tuple(padding)
        )

    def forward(self, x: Tensor) -> Tensor:
        return pad(x, self.padding, mode="reflect")

    def extra_repr(self) -> str:
        return f"padding={self.padding}"


class ReflectionPad2d(Module):
    """Pad a 4-D tensor using reflection."""

    def __init__(self, padding: int | tuple[int, int, int, int]) -> None:
        super().__init__()
        self.padding = (padding,) * 4 if isinstance(padding, int) else tuple(padding)

    def forward(self, x: Tensor) -> Tensor:
        return pad(x, self.padding, mode="reflect")

    def extra_repr(self) -> str:
        return f"padding={self.padding}"


class ReplicationPad1d(Module):
    """Pad a 3-D tensor using replication."""

    def __init__(self, padding: _Size2d) -> None:
        super().__init__()
        self.padding = (
            (padding, padding) if isinstance(padding, int) else tuple(padding)
        )

    def forward(self, x: Tensor) -> Tensor:
        return pad(x, self.padding, mode="replicate")

    def extra_repr(self) -> str:
        return f"padding={self.padding}"


class ReplicationPad2d(Module):
    """Pad a 4-D tensor using replication."""

    def __init__(self, padding: int | tuple[int, int, int, int]) -> None:
        super().__init__()
        self.padding = (padding,) * 4 if isinstance(padding, int) else tuple(padding)

    def forward(self, x: Tensor) -> Tensor:
        return pad(x, self.padding, mode="replicate")

    def extra_repr(self) -> str:
        return f"padding={self.padding}"


class ReplicationPad3d(Module):
    """Pad a 5-D tensor using replication."""

    def __init__(self, padding: int | tuple[int, int, int, int, int, int]) -> None:
        super().__init__()
        self.padding = (padding,) * 6 if isinstance(padding, int) else tuple(padding)

    def forward(self, x: Tensor) -> Tensor:
        return pad(x, self.padding, mode="replicate")

    def extra_repr(self) -> str:
        return f"padding={self.padding}"
