"""Unit tests for the final-2% top-level closure: ``histogram2d`` /
``histogramdd``, DLPack interop (``from_dlpack`` / ``to_dlpack`` plus
``Tensor.__dlpack__`` / ``__dlpack_device__``), and ``poisson`` sampling.
"""

import numpy as np
import pytest

import lucid


# ── histogram2d / histogramdd ─────────────────────────────────────────────


class TestHistogram2d:
    def test_basic_shapes(self) -> None:
        x = lucid.tensor(np.random.RandomState(0).randn(100).astype(np.float32))
        y = lucid.tensor(np.random.RandomState(1).randn(100).astype(np.float32))
        counts, xe, ye = lucid.histogram2d(x, y, bins=4)
        assert counts.shape == (4, 4)
        assert xe.shape == (5,)
        assert ye.shape == (5,)
        assert int(counts.sum().item()) == 100

    def test_explicit_range_and_unequal_bins(self) -> None:
        x = lucid.tensor([0.5, 1.5, 2.5, 3.5])
        y = lucid.tensor([0.0, 1.0, 2.0, 3.0])
        counts, xe, ye = lucid.histogram2d(
            x, y, bins=(2, 3), range=((0.0, 4.0), (0.0, 3.0))
        )
        assert counts.shape == (2, 3)
        assert xe.shape == (3,)
        assert ye.shape == (4,)

    def test_density_normalisation(self) -> None:
        # With density=True, ``integral over cells = fraction in range``;
        # equals 1 only if all samples fall inside.  Use uniform samples
        # so we can guarantee containment.
        rng = np.random.RandomState(2)
        x = lucid.tensor(rng.uniform(-1.0, 1.0, size=500).astype(np.float32))
        y = lucid.tensor(rng.uniform(-1.0, 1.0, size=500).astype(np.float32))
        counts, xe, ye = lucid.histogram2d(
            x, y, bins=8, range=((-1.0, 1.0), (-1.0, 1.0)), density=True
        )
        dx = (xe[1] - xe[0]).item()
        dy = (ye[1] - ye[0]).item()
        integral = float(counts.sum().item()) * dx * dy
        assert abs(integral - 1.0) < 1e-4


class TestHistogramdd:
    def test_3d_shapes(self) -> None:
        data = lucid.tensor(np.random.RandomState(4).randn(50, 3).astype(np.float32))
        counts, edges = lucid.histogramdd(data, bins=3)
        assert counts.shape == (3, 3, 3)
        assert len(edges) == 3
        for e in edges:
            assert e.shape == (4,)
        assert int(counts.sum().item()) == 50

    def test_per_axis_bins(self) -> None:
        data = lucid.tensor(np.random.RandomState(5).randn(40, 2).astype(np.float32))
        counts, edges = lucid.histogramdd(data, bins=[5, 7])
        assert counts.shape == (5, 7)
        assert edges[0].shape == (6,)
        assert edges[1].shape == (8,)

    def test_non_2d_input_rejected(self) -> None:
        with pytest.raises(ValueError):
            lucid.histogramdd(lucid.tensor([1.0, 2.0, 3.0]))

    def test_bins_length_mismatch_rejected(self) -> None:
        data = lucid.tensor(np.zeros((10, 3), dtype=np.float32))
        with pytest.raises(ValueError):
            lucid.histogramdd(data, bins=[5, 7])  # length 2 vs ndim 3.


# ── DLPack interop ────────────────────────────────────────────────────────


class TestDLPack:
    def test_to_dlpack_returns_capsule(self) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0])
        caps = lucid.to_dlpack(t)
        assert type(caps).__name__ == "PyCapsule"

    def test_dlpack_protocol_methods(self) -> None:
        t = lucid.tensor([1.0, 2.0])
        # __dlpack__ returns a PyCapsule.
        assert type(t.__dlpack__()).__name__ == "PyCapsule"
        # __dlpack_device__ reports CPU (1, 0).
        assert t.__dlpack_device__() == (1, 0)

    def test_round_trip_through_numpy(self) -> None:
        t = lucid.tensor([1.5, 2.5, 3.5])
        # Modern path: pass tensor object to np.from_dlpack.
        arr = np.from_dlpack(t)
        np.testing.assert_array_equal(arr, [1.5, 2.5, 3.5])

    def test_from_dlpack_consumes_numpy(self) -> None:
        arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        t = lucid.from_dlpack(arr)
        np.testing.assert_array_equal(t.numpy(), [10.0, 20.0, 30.0])

    def test_metal_tensor_exports_via_cpu(self) -> None:
        # GPU tensors round-trip through CPU; the consumed array still
        # carries the original values.
        g = lucid.tensor([1.0, 2.0], device="metal")
        arr = np.from_dlpack(g)
        np.testing.assert_array_equal(arr, [1.0, 2.0])


# ── poisson sampling ──────────────────────────────────────────────────────


class TestPoisson:
    def test_zero_rate_always_zero(self) -> None:
        out = lucid.poisson(lucid.tensor([0.0, 0.0, 0.0])).numpy()
        np.testing.assert_array_equal(out, [0, 0, 0])

    def test_negative_rate_rejected(self) -> None:
        with pytest.raises(ValueError):
            lucid.poisson(lucid.tensor([-1.0]))

    def test_shape_and_dtype_preserved(self) -> None:
        rates = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = lucid.poisson(rates)
        assert out.shape == rates.shape
        assert out.dtype == lucid.int64

    def test_mean_matches_rate_small(self) -> None:
        # Small-rate regime uses Knuth — Poisson identity E[X] = λ.
        lucid.manual_seed(0)
        rates = lucid.tensor([5.0])
        N = 2000
        samples = np.array([
            float(lucid.poisson(rates).item()) for _ in range(N)
        ])
        assert abs(samples.mean() - 5.0) < 0.3  # within ~3·SE.

    def test_mean_matches_rate_large(self) -> None:
        # Large-rate regime uses Normal approx.
        lucid.manual_seed(0)
        rates = lucid.tensor([100.0])
        N = 500
        samples = np.array([
            float(lucid.poisson(rates).item()) for _ in range(N)
        ])
        assert abs(samples.mean() - 100.0) < 2.0  # within ~3·SE.
        assert abs(samples.std() - 10.0) < 1.5    # √100 = 10.

    def test_reproducible_with_manual_seed(self) -> None:
        lucid.manual_seed(7)
        a = lucid.poisson(lucid.tensor([3.0, 3.0, 3.0])).numpy()
        lucid.manual_seed(7)
        b = lucid.poisson(lucid.tensor([3.0, 3.0, 3.0])).numpy()
        np.testing.assert_array_equal(a, b)

    def test_explicit_generator_isolates_stream(self) -> None:
        g1 = lucid.Generator(seed=1)
        g2 = lucid.Generator(seed=2)
        rates = lucid.tensor([5.0] * 20)
        a = lucid.poisson(rates, generator=g1).numpy()
        b = lucid.poisson(rates, generator=g2).numpy()
        # Different seeds — overwhelmingly unlikely to match exactly.
        assert not np.array_equal(a, b)
