"""Reduction ops — sum / mean / max / min / argmax / cumsum / std / var / etc."""

import numpy as np
import pytest

import lucid

# ── basic full-tensor reductions ────────────────────────────────────────


class TestSum:
    def test_full(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        assert lucid.sum(t).item() == 10.0

    def test_along_dim(self, device: str) -> None:
        t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        np.testing.assert_array_equal(lucid.sum(t, dim=0).numpy(), [4.0, 6.0])
        np.testing.assert_array_equal(lucid.sum(t, dim=1).numpy(), [3.0, 7.0])

    def test_keepdim(self, device: str) -> None:
        t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        out = lucid.sum(t, dim=0, keepdim=True)
        assert out.shape == (1, 2)


class TestMean:
    def test_full(self, device: str) -> None:
        t = lucid.tensor([2.0, 4.0, 6.0], device=device)
        assert lucid.mean(t).item() == 4.0

    def test_along_dim(self, device: str) -> None:
        t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        np.testing.assert_array_equal(lucid.mean(t, dim=0).numpy(), [2.0, 3.0])


class TestMax:
    def test_full(self, device: str) -> None:
        t = lucid.tensor([3.0, 1.0, 5.0, 2.0], device=device)
        assert lucid.max(t).item() == 5.0

    def test_along_dim(self, device: str) -> None:
        t = lucid.tensor([[1.0, 4.0], [3.0, 2.0]], device=device)
        np.testing.assert_array_equal(lucid.max(t, dim=0).numpy(), [3.0, 4.0])


class TestMin:
    def test_full(self, device: str) -> None:
        t = lucid.tensor([3.0, 1.0, 5.0, 2.0], device=device)
        assert lucid.min(t).item() == 1.0


class TestProd:
    def test_full(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        assert lucid.prod(t).item() == 24.0


class TestArgmax:
    def test_full(self, device: str) -> None:
        t = lucid.tensor([3.0, 1.0, 5.0, 2.0], device=device)
        assert int(lucid.argmax(t).item()) == 2

    def test_along_dim(self, device: str) -> None:
        t = lucid.tensor([[1.0, 4.0], [3.0, 2.0]], device=device)
        out = lucid.argmax(t, dim=0).numpy()
        np.testing.assert_array_equal(out, [1, 0])


class TestArgmin:
    def test_full(self, device: str) -> None:
        t = lucid.tensor([3.0, 1.0, 5.0, 2.0], device=device)
        assert int(lucid.argmin(t).item()) == 1


class TestCumsum:
    def test_basic(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        np.testing.assert_array_equal(
            lucid.cumsum(t, dim=0).numpy(), [1.0, 3.0, 6.0, 10.0]
        )


class TestCumprod:
    def test_basic(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        np.testing.assert_array_equal(
            lucid.cumprod(t, dim=0).numpy(), [1.0, 2.0, 6.0, 24.0]
        )


class TestStdVar:
    def test_var(self, device: str) -> None:
        # population variance of [0, 1, 2] = 2/3.
        t = lucid.tensor([0.0, 1.0, 2.0], device=device)
        v = lucid.var(t, correction=0).item()
        assert abs(v - 2.0 / 3.0) < 1e-5

    def test_std(self, device: str) -> None:
        t = lucid.tensor([0.0, 1.0, 2.0], device=device)
        s = lucid.std(t, correction=0).item()
        assert abs(s - np.sqrt(2.0 / 3.0)) < 1e-5

    def test_var_corrected(self, device: str) -> None:
        # Bessel-corrected variance of [0, 1, 2] = 1.
        t = lucid.tensor([0.0, 1.0, 2.0], device=device)
        v = lucid.var(t, correction=1).item()
        assert abs(v - 1.0) < 1e-5


class TestTrace:
    def test_basic(self, device: str) -> None:
        t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        assert lucid.trace(t).item() == 5.0


class TestLogsumexp:
    def test_known(self, device: str) -> None:
        # logsumexp([0, 0, 0]) = log 3.
        t = lucid.tensor([0.0, 0.0, 0.0], device=device)
        assert abs(lucid.logsumexp(t, dim=0).item() - np.log(3.0)) < 1e-5


# ── nan-safe reductions ─────────────────────────────────────────────────


class TestNanReductions:
    def test_nansum(self, device: str) -> None:
        t = lucid.tensor([1.0, float("nan"), 3.0], device=device)
        assert lucid.nansum(t).item() == 4.0

    def test_nanmean(self, device: str) -> None:
        t = lucid.tensor([2.0, float("nan"), 4.0], device=device)
        assert lucid.nanmean(t).item() == 3.0


# ── shape coverage across reduction axes ────────────────────────────────


@pytest.mark.parametrize("shape", [(2, 3, 4), (5,), (1, 1, 1)])
@pytest.mark.parametrize("dim", [0, -1])
def test_sum_shape_inference(shape: tuple[int, ...], dim: int, device: str) -> None:
    t = lucid.zeros(*shape, device=device)
    out = lucid.sum(t, dim=dim)
    expected_shape = list(shape)
    expected_shape.pop(dim if dim >= 0 else dim + len(shape))
    assert out.shape == tuple(expected_shape)
