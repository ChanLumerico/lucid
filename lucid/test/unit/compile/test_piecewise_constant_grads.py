"""A compiled step gives a parameter behind a step function a zero gradient.

``floor`` / ``ceil`` / ``round`` / ``trunc`` / ``sign`` are piecewise
constant, and eager returns zeros shaped like the input for them.  The
compiled walk used to treat them as grad sinks, so a parameter reached
only through one had no gradient at all — and the engine then filled that
``None`` from stale memory.  An infinite gradient just downstream
(``erfinv`` at ±1) must not turn the zero into NaN either.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn

from lucid.test.unit.compile._helpers import COMPILE_DEVICE

_OPS = {
    "floor": lucid.floor,
    "ceil": lucid.ceil,
    "round": lucid.round,
    "trunc": lucid.trunc,
    "sign": lucid.sign,
}


class _Behind(nn.Module):
    def __init__(self, op: object, then: object) -> None:
        super().__init__()
        self.w = nn.Parameter(lucid.linspace(-1.5, 1.5, 12).reshape(3, 4))
        self._op = op
        self._then = then

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self._then(self._op((x * self.w).clamp(-1, 1))).sum()


def _compiled_and_eager(model: nn.Module) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(COMPILE_DEVICE)
    x = lucid.full((3, 4), 1.25).to(COMPILE_DEVICE)
    model(x).backward()
    want = model.w.grad.numpy().copy()
    model.w.grad = None
    step = lucid.compile.make_step(model, lambda out: out)
    step(x).backward()
    assert not step.eager_only, "fell back to eager"
    # Present in the executable, not a left-out slot the engine fills.
    assert not any(e.absent for e in step.cache.values()), "w's gradient left out"
    assert model.w.grad is not None, "compiled step left w.grad empty"
    return model.w.grad.numpy(), want


@pytest.mark.parametrize("name", sorted(_OPS))
def test_the_compiled_gradient_is_eagers_zero(name: str) -> None:
    got, want = _compiled_and_eager(_Behind(_OPS[name], lambda t: t * 3.0))
    np.testing.assert_array_equal(want, np.zeros_like(want))
    np.testing.assert_array_equal(got, want)


def test_an_infinite_gradient_downstream_does_not_make_it_nan() -> None:
    got, want = _compiled_and_eager(_Behind(lucid.round, lucid.erfinv))
    assert not np.isnan(got).any()
    np.testing.assert_array_equal(got, want)
