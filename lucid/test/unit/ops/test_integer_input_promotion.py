"""Float-natured ops on integer tensors answer in float, not truncated.

``mean`` of ``[1, 2, 3, 4]`` was 2, ``int @ float`` cast the float operand to
int, and an in-place float op on an integer tensor (``exp_``) kept the
integer's 8-byte strides over the new 4-byte floats, reading every other
element — all without an error.
"""

import math

import pytest

import lucid

_DEVICES = [
    "cpu",
    pytest.param(
        "metal",
        marks=pytest.mark.skipif(
            not lucid.metal.is_available(), reason="no Metal device"
        ),
    ),
]


def _ints(*values: int, device: str = "cpu") -> lucid.Tensor:
    return lucid.tensor(list(values), dtype=lucid.int64, device=device)


@pytest.mark.parametrize("device", _DEVICES)
def test_mean_of_integers_is_floating(device: str) -> None:
    out = _ints(1, 2, 3, 4, device=device).mean()
    assert out.dtype == lucid.float32
    assert out.item() == 2.5


@pytest.mark.parametrize("device", _DEVICES)
def test_norm_of_integers_is_floating(device: str) -> None:
    out = lucid.linalg.norm(_ints(1, 1, device=device))
    assert out.item() == pytest.approx(math.sqrt(2))


@pytest.mark.parametrize("device", _DEVICES)
def test_matmul_promotes_an_integer_operand(device: str) -> None:
    a = lucid.tensor([[1, 2]], dtype=lucid.int64, device=device)
    b = lucid.tensor([[0.5], [0.25]], device=device)
    assert (a @ b).dtype == lucid.float32
    assert (a @ b).tolist() == [[1.0]]
    assert lucid.matmul(a, b).tolist() == [[1.0]]
    assert (b.reshape(1, 2) @ a.reshape(2, 1)).tolist() == [[1.0]]


@pytest.mark.parametrize("device", _DEVICES)
def test_in_place_float_op_on_integers_reads_back_right(device: str) -> None:
    x = _ints(1, 2, 3, 4, 5, device=device)
    x.exp_()
    assert x.dtype == lucid.float32
    assert x.tolist() == pytest.approx([math.exp(v) for v in range(1, 6)], rel=1e-6)
