"""``jvp`` / ``hvp`` / ``vhp`` are probed with ``v`` in the input space.

``vjp``'s vector is shaped like the function's output; the other three
take one shaped like its input.  The audit shaped every ``v`` like an
output, so a probe of ``hvp`` would be refused for the probe's mistake
and read as a defect in the op.  These guard the instrument by name, so
the rule holds whether or not a given function is on the surface yet.
"""

from collections.abc import Callable
from typing import Any

import pytest

from lucid._tensor.tensor import Tensor
from lucid.test.audit import _autospec, _specs


def _shapes(value: Any) -> list[tuple[int, ...]]:
    items = value if isinstance(value, list | tuple) else [value]
    return [tuple(item.shape) for item in items]


def _required_v(
    func: Callable[..., Tensor], inputs: Tensor | tuple[Tensor, ...], v: Tensor
) -> Tensor:
    del func, inputs
    return v


@pytest.mark.parametrize("op", ["jvp", "hvp", "vhp"])
def test_a_required_v_is_shaped_like_the_inputs(op: str) -> None:
    call = next(_autospec.invocations(_required_v, op, "moderate"))
    _, inputs, v = call.args
    assert _shapes(v) == _shapes(inputs)


def test_a_vjp_cotangent_is_shaped_like_the_scalar_output() -> None:
    call = next(_autospec.invocations(_required_v, "vjp", "moderate"))
    assert _shapes(call.args[2]) == [()]


@pytest.mark.parametrize("op", ["jvp", "hvp", "vhp"])
def test_the_hand_spec_gives_v_the_inputs_shape(op: str) -> None:
    call = next(_specs.invocations(op, "moderate", qualname=f"lucid.autograd.{op}"))
    _, inputs, v = call.args
    assert _shapes(v) == _shapes(inputs)
    assert call.same_shape == (2,)
