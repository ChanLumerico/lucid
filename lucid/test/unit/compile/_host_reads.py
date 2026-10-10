"""Every way a traced function hands a tensor's values to Python.

Shared by the compile and Core ML tests of LCD-337: each entry reads a
value computed from the input and builds the result from it, so a trace
that kept the value it read would answer every later input with the
first one's result.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn


def _numpy_total(t: lucid.Tensor) -> float:
    pytest.importorskip("numpy")
    return float(t.numpy().sum())


HOST_READS: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "item": lambda x: x * 0 + lucid.full_like(x, x.sum().item()),
    "tolist": lambda x: x * 0 + x.sum().tolist(),
    "numpy": lambda x: x * 0 + _numpy_total(x.sum()),
    "bool": lambda x: x * 2 if bool(x.sum() > 0) else x * 3,
    "if": lambda x: x * 2 if x.sum() > 0 else x * 3,
    "float": lambda x: x * 0 + float(x.sum()),
    "int": lambda x: x * 0 + int(x.sum()),
    # The argument itself, read before any operation has consumed it.
    "argument-tolist": lambda x: lucid.full_like(x, sum(x.tolist())) + x * 0,
    "argument-numpy": lambda x: lucid.full_like(x, _numpy_total(x)) + x * 0,
}


class Apply(nn.Module):
    def __init__(self, fn: Callable[[lucid.Tensor], lucid.Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn(x)


def inputs() -> tuple[lucid.Tensor, lucid.Tensor]:
    """Two inputs whose sums differ in sign, so every entry's answer moves."""
    lucid.manual_seed(0)
    first = -lucid.randn(16).abs() - 1
    second = lucid.randn(16).abs() * 7 + 3
    return first, second
