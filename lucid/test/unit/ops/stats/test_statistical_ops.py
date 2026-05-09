"""Statistics-flavored ops: quantile / cov / corrcoef / cdist /
histogram* / multinomial / poisson."""

import numpy as np
import pytest

import lucid
from lucid.test._helpers.compare import assert_close


class TestQuantile:
    def test_median(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
        out = lucid.quantile(t, 0.5).item()
        assert abs(out - 3.0) < 1e-6

    def test_p25(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
        out = lucid.quantile(t, 0.25).item()
        assert abs(out - 2.0) < 1e-6


class TestCov:
    def test_basic(self, device: str) -> None:
        # Two-row matrix: (3, 4) — covariance is a 3×3 matrix.
        x = lucid.tensor(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], [0.0, 1.0, 0.0, 1.0]],
            device=device,
        )
        out = lucid.cov(x).numpy()
        # Variances on the diagonal must be non-negative.
        assert np.all(np.diag(out) >= 0)


class TestCorrcoef:
    def test_perfect_positive(self, device: str) -> None:
        # Two perfectly-correlated rows.
        x = lucid.tensor(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]],
            device=device,
        )
        out = lucid.corrcoef(x).numpy()
        assert abs(out[0, 1] - 1.0) < 1e-3


class TestCdist:
    def test_l2(self, device: str) -> None:
        a = lucid.tensor([[0.0, 0.0]], device=device)
        b = lucid.tensor([[3.0, 4.0]], device=device)
        out = lucid.cdist(a, b).item()
        assert abs(out - 5.0) < 1e-4


class TestBincount:
    def test_basic(self, device: str) -> None:
        t = lucid.tensor([0, 1, 1, 2, 2, 2, 3], dtype=lucid.int64, device=device)
        out = lucid.bincount(t).numpy()
        np.testing.assert_array_equal(out, [1, 2, 3, 1])


class TestHistogram:
    # ``lucid.histogram`` has a known parameter-shadowing issue when
    # ``range=(lo, hi)`` is passed positionally — work around by
    # passing explicit bin edges via the ``bins`` argument.

    def test_basic(self, device: str) -> None:
        t = lucid.tensor([0.5, 1.5, 1.5, 2.5], device=device)
        counts, edges = lucid.histogram(t, bins=[0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(counts.numpy().astype(np.int32), [1, 2, 1])

    def test_density(self, device: str) -> None:
        t = lucid.tensor([0.5, 1.5, 2.5, 0.5], device=device)
        counts, edges = lucid.histogram(t, bins=[0.0, 1.0, 2.0, 3.0], density=True)
        bw = edges[1].item() - edges[0].item()
        assert abs(counts.sum().item() * bw - 1.0) < 1e-5


class TestHistogram2d:
    def test_basic(self, device: str) -> None:
        x = lucid.tensor([0.0, 1.0, 1.0, 2.0], device=device)
        y = lucid.tensor([0.0, 0.0, 1.0, 1.0], device=device)
        counts, xe, ye = lucid.histogram2d(x, y, bins=2, range=((0.0, 2.0), (0.0, 1.0)))
        assert counts.shape == (2, 2)
        assert xe.shape == (3,)
        assert ye.shape == (3,)


class TestHistogramdd:
    def test_3d(self, device: str) -> None:
        np.random.seed(0)
        data_np = np.random.uniform(-1.0, 1.0, size=(50, 3)).astype(np.float32)
        data = lucid.tensor(data_np.copy(), device=device)
        counts, edges = lucid.histogramdd(data, bins=3)
        assert counts.shape == (3, 3, 3)
        assert int(counts.sum().item()) == 50


class TestMultinomial:
    def test_shape(self, device: str) -> None:
        probs = lucid.tensor([0.2, 0.3, 0.5], device=device)
        out = lucid.multinomial(probs, num_samples=10, replacement=True)
        assert out.shape == (10,)
        arr = out.numpy()
        assert (arr >= 0).all() and (arr < 3).all()


class TestPoissonOp:
    def test_zero_rate(self, device: str) -> None:
        rates = lucid.tensor([0.0, 0.0, 0.0], device=device)
        out = lucid.poisson(rates).numpy()
        np.testing.assert_array_equal(out, [0, 0, 0])

    def test_positive_samples(self, device: str) -> None:
        lucid.manual_seed(0)
        rates = lucid.tensor([5.0] * 200, device=device)
        out = lucid.poisson(rates).numpy()
        assert (out >= 0).all()
        # Mean should be near 5 with this many samples.
        assert abs(out.mean() - 5.0) < 1.0
