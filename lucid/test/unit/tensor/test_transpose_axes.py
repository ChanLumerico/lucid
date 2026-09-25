"""``transpose`` takes the reference framework's two axes, or none.

The engine op takes no axes and reverses them all, as NumPy's does, so
``x.transpose(0, 1)`` — the reference's only form — raised a TypeError.
Its docstring meanwhile said the no-axis form swapped the last two; it
reverses them all, as ``.T`` does.
"""

import numpy as np
import pytest

import lucid

DEVICES = ["cpu", "metal"]
_A = np.arange(24, dtype=np.float32).reshape(2, 3, 4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dims", [(0, 1), (1, 2), (0, -1), (-1, -2)], ids=str)
def test_two_axes_are_swapped(device: str, dims: tuple[int, int]) -> None:
    x = lucid.tensor(_A, device=device)
    want = np.swapaxes(_A, *dims)
    np.testing.assert_array_equal(x.transpose(*dims).numpy(), want)
    np.testing.assert_array_equal(lucid.transpose(x, *dims).numpy(), want)


@pytest.mark.parametrize("device", DEVICES)
def test_no_axes_reverses_them_all(device: str) -> None:
    x = lucid.tensor(_A, device=device)
    np.testing.assert_array_equal(x.transpose().numpy(), _A.transpose())
    np.testing.assert_array_equal(x.transpose().numpy(), x.T.numpy())


def test_one_axis_is_refused() -> None:
    with pytest.raises(TypeError, match="no axes, or two"):
        lucid.tensor(_A).transpose(0)


def test_the_swap_is_differentiable() -> None:
    x = lucid.tensor(_A, requires_grad=True)
    (x.transpose(0, 2) * lucid.tensor(np.swapaxes(_A, 0, 2))).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), _A)
