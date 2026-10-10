"""Every way a traced function hands a tensor's values to Python.

Shared by the compile and Core ML tests of LCD-337: each entry reads a
value the result depends on — computed from the input, the input itself,
or the model's own parameter or buffer — and builds the result from it,
so a trace that kept the value it read would answer every later call
with the first one's result.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn


class Apply(nn.Module):
    """``fn(self, x)``, with a parameter and a buffer the test may change."""

    def __init__(self, fn: Callable[[Apply, lucid.Tensor], lucid.Tensor]) -> None:
        super().__init__()
        self.fn = fn
        self.scale = nn.Parameter(lucid.ones(()))
        self.register_buffer("offset", lucid.zeros(()))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn(self, x)

    def change_state(self) -> None:
        """New weights, as an optimizer step or ``load_state_dict`` gives."""
        with lucid.no_grad():
            self.scale.copy_(lucid.full_like(self.scale, 5.0))
            offset = self.get_buffer("offset")
            offset.copy_(lucid.full_like(offset, 7.0))


def _numpy_total(t: lucid.Tensor) -> float:
    pytest.importorskip("numpy")
    return float(t.numpy().sum())


HOST_READS: dict[str, Callable[[Apply, lucid.Tensor], lucid.Tensor]] = {
    "item": lambda m, x: x * 0 + lucid.full_like(x, x.sum().item()),
    "tolist": lambda m, x: x * 0 + x.sum().tolist(),
    "numpy": lambda m, x: x * 0 + _numpy_total(x.sum()),
    "bool": lambda m, x: x * 2 if bool(x.sum() > 0) else x * 3,
    "if": lambda m, x: x * 2 if x.sum() > 0 else x * 3,
    "float": lambda m, x: x * 0 + float(x.sum()),
    "int": lambda m, x: x * 0 + int(x.sum()),
    # The argument itself, read before any operation has consumed it.
    "argument-tolist": lambda m, x: lucid.full_like(x, sum(x.tolist())) + x * 0,
    "argument-numpy": lambda m, x: lucid.full_like(x, _numpy_total(x)) + x * 0,
    # The model's own state, read before any operation has consumed it. A
    # Core ML package holds its weights as constants, so only the compile
    # path, which feeds them on every run, must not keep the value read.
    "parameter-item": lambda m, x: x * m.scale.item(),
    "buffer-item": lambda m, x: x + m.get_buffer("offset").item(),
}

#: Entries whose read is of the model's weights rather than of its input.
STATE_READS = frozenset({"parameter-item", "buffer-item"})


def _made_on_the_spot(m: Apply, x: lucid.Tensor) -> lucid.Tensor:
    return x * lucid.tensor([2.0, 3.0]).tolist()[1]


def _validation_only(m: Apply, x: lucid.Tensor) -> lucid.Tensor:
    with lucid.compile._host_check():
        if bool((x < -1e6).any()):
            raise ValueError("unreachable on these inputs")
    return x * 3


def _advanced_index(m: Apply, x: lucid.Tensor) -> lucid.Tensor:
    # Two coordinate indices computed from the input: the CPU checks each
    # against its own dim by reading traced values — only to decide whether
    # to raise (LCD-336).  ``x[rows, cols]`` on the 4 x 4 view is ``x``.
    n = x.shape[0]
    zero = (x * 0).long()
    rows = zero + lucid.tensor([i // 4 for i in range(n)], device=x.device)
    cols = zero + lucid.tensor([i % 4 for i in range(n)], device=x.device)
    return x.reshape(n // 4, 4)[rows, cols] * 3


#: Reads that leave the trace whole: each answers ``x * 3``. A tensor built
#: from Python numbers holds no input data, and a read under
#: ``lucid.compile._host_check()`` only decides whether to raise.
CONSTANT_READS: dict[str, Callable[[Apply, lucid.Tensor], lucid.Tensor]] = {
    "made-on-the-spot": _made_on_the_spot,
    "validation-only": _validation_only,
    "advanced-index-bounds": _advanced_index,
}


def inputs() -> tuple[lucid.Tensor, lucid.Tensor]:
    """Two inputs whose sums differ in sign, so every entry's answer moves."""
    lucid.manual_seed(0)
    first = -lucid.randn(16).abs() - 1
    second = lucid.randn(16).abs() * 7 + 3
    return first, second
