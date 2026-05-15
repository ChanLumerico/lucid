"""Tensor factory functions — value, shape, dtype, device coverage."""

import numpy as np

import lucid
from lucid.test._fixtures.devices import skip_if_unsupported
from lucid.test._helpers.compare import assert_close, assert_equal_int

# ── deterministic factories ──────────────────────────────────────────────


class TestZeros:
    def test_shape(self, device: str) -> None:
        t = lucid.zeros(3, 4, device=device)
        assert t.shape == (3, 4)

    def test_values(self, device: str, float_dtype: lucid.dtype) -> None:
        skip_if_unsupported(device, float_dtype)
        t = lucid.zeros(2, 3, dtype=float_dtype, device=device)
        assert_equal_int(t, np.zeros((2, 3)))

    def test_dtype_propagates(self, device: str) -> None:
        for dt in (lucid.float32, lucid.float64, lucid.int32, lucid.int64):
            if device == "metal" and dt == lucid.float64:
                continue  # metal can't do f64; covered on CPU.
            t = lucid.zeros(2, dtype=dt, device=device)
            assert t.dtype == dt


class TestOnes:
    def test_values(self, device: str, float_dtype: lucid.dtype) -> None:
        skip_if_unsupported(device, float_dtype)
        t = lucid.ones(2, 3, dtype=float_dtype, device=device)
        assert_close(t, np.ones((2, 3)))

    def test_zero_dim(self, device: str) -> None:
        t = lucid.ones(device=device)
        assert t.shape == ()
        assert_close(t, np.array(1.0))


class TestEmpty:
    def test_shape_only(self, device: str) -> None:
        t = lucid.empty(4, 5, device=device)
        assert t.shape == (4, 5)

    def test_dtype(self, device: str) -> None:
        skip_if_unsupported(device, lucid.float64)
        t = lucid.empty(3, dtype=lucid.float64, device=device)
        assert t.dtype == lucid.float64


class TestFull:
    def test_known_value(self, device: str, float_dtype: lucid.dtype) -> None:
        skip_if_unsupported(device, float_dtype)
        t = lucid.full((2, 3), 7.5, dtype=float_dtype, device=device)
        assert_close(t, np.full((2, 3), 7.5))

    def test_int_value(self, device: str) -> None:
        t = lucid.full((4,), -3, dtype=lucid.int64, device=device)
        assert_equal_int(t, np.full(4, -3, dtype=np.int64))


class TestEye:
    def test_square_identity(self, device: str, float_dtype: lucid.dtype) -> None:
        skip_if_unsupported(device, float_dtype)
        t = lucid.eye(4, dtype=float_dtype, device=device)
        assert_close(t, np.eye(4))

    def test_rectangular(self, device: str) -> None:
        t = lucid.eye(3, 5, device=device)
        expected = np.eye(3, 5)
        assert_close(t, expected)


class TestArange:
    def test_basic(self, device: str) -> None:
        t = lucid.arange(0.0, 5.0, 1.0, device=device)
        assert_close(t, np.arange(0.0, 5.0, 1.0))

    def test_negative_step(self, device: str) -> None:
        t = lucid.arange(5, -1, -1, dtype=lucid.int32, device=device)
        assert_equal_int(t, np.arange(5, -1, -1, dtype=np.int32))

    def test_float_step(self, device: str) -> None:
        t = lucid.arange(0.0, 1.0, 0.25, device=device)
        assert_close(t, np.arange(0.0, 1.0, 0.25))


class TestLinspace:
    def test_endpoint(self, device: str, float_dtype: lucid.dtype) -> None:
        skip_if_unsupported(device, float_dtype)
        t = lucid.linspace(0.0, 1.0, 5, dtype=float_dtype, device=device)
        assert_close(t, np.linspace(0.0, 1.0, 5))

    def test_single_point(self, device: str) -> None:
        t = lucid.linspace(3.0, 7.0, 1, device=device)
        assert t.shape == (1,)


class TestLogspace:
    def test_known(self, device: str, float_dtype: lucid.dtype) -> None:
        skip_if_unsupported(device, float_dtype)
        t = lucid.logspace(0.0, 2.0, 3, dtype=float_dtype, device=device)
        # Default base is 10 — [1, 10, 100].
        assert_close(t, np.logspace(0.0, 2.0, 3), atol=1e-4)


class TestZerosLike:
    def test_shape_dtype_device(self, device: str) -> None:
        src = lucid.tensor([1.0, 2.0, 3.0], device=device)
        t = lucid.zeros_like(src)
        assert t.shape == src.shape
        assert t.dtype == src.dtype
        assert_close(t, np.zeros(3))


class TestOnesLike:
    def test_shape_dtype_device(self, device: str) -> None:
        src = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        t = lucid.ones_like(src)
        assert t.shape == src.shape
        assert_close(t, np.ones((2, 2)))


class TestFullLike:
    def test_value(self, device: str) -> None:
        src = lucid.zeros(3, 4, device=device)
        t = lucid.full_like(src, 9.0)
        assert_close(t, np.full((3, 4), 9.0))


# ── random factories ─────────────────────────────────────────────────────


class TestRand:
    def test_in_unit_interval(self, device: str, float_dtype: lucid.dtype) -> None:
        skip_if_unsupported(device, float_dtype)
        t = lucid.rand(64, dtype=float_dtype, device=device)
        arr = t.numpy()
        assert (arr >= 0).all() and (arr < 1).all()

    def test_shape(self, device: str) -> None:
        assert lucid.rand(2, 3, 4, device=device).shape == (2, 3, 4)

    def test_reproducible_with_manual_seed(self, device: str) -> None:
        lucid.manual_seed(0)
        a = lucid.rand(8, device=device).numpy()
        lucid.manual_seed(0)
        b = lucid.rand(8, device=device).numpy()
        np.testing.assert_array_equal(a, b)


class TestRandn:
    def test_shape(self, device: str) -> None:
        assert lucid.randn(5, 3, device=device).shape == (5, 3)

    def test_distribution_loose(self, device: str) -> None:
        lucid.manual_seed(0)
        arr = lucid.randn(10_000, device=device).numpy()
        assert abs(arr.mean()) < 0.1
        assert abs(arr.std() - 1.0) < 0.1


class TestRandint:
    def test_in_range(self, device: str) -> None:
        t = lucid.randint(low=0, high=10, size=(64,), device=device)
        arr = t.numpy()
        assert (arr >= 0).all() and (arr < 10).all()

    def test_dtype_int64_default(self, device: str) -> None:
        t = lucid.randint(low=0, high=2, size=(4,), device=device)
        assert t.dtype == lucid.int64


class TestNormal:
    def test_shape(self, device: str) -> None:
        t = lucid.normal(mean=0.0, std=1.0, size=(3, 4), device=device)
        assert t.shape == (3, 4)

    def test_loose_moments(self, device: str) -> None:
        lucid.manual_seed(0)
        arr = lucid.normal(mean=2.0, std=0.5, size=(10_000,), device=device).numpy()
        assert abs(arr.mean() - 2.0) < 0.05
        assert abs(arr.std() - 0.5) < 0.05


class TestBernoulli:
    def test_binary(self, device: str) -> None:
        t = lucid.bernoulli(0.5, size=(100,), device=device)
        arr = t.numpy()
        assert set(arr.flatten().tolist()) <= {0.0, 1.0}


class TestRandperm:
    def test_is_permutation(self, device: str) -> None:
        n = 16
        t = lucid.randperm(n, device=device)
        arr = sorted(t.numpy().tolist())
        assert arr == list(range(n))


class TestRandLike:
    def test_inherits_metadata(self, device: str) -> None:
        src = lucid.zeros(3, 4, device=device)
        t = lucid.rand_like(src)
        assert t.shape == src.shape
        assert t.device == src.device


class TestRandnLike:
    def test_inherits_metadata(self, device: str) -> None:
        src = lucid.zeros(2, 5, device=device)
        t = lucid.randn_like(src)
        assert t.shape == src.shape


# ── tensor() / as_tensor() / from_numpy() ───────────────────────────────


class TestTensorFactory:
    def test_from_list(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0], device=device)
        assert_close(t, np.array([1.0, 2.0, 3.0]))

    def test_from_nested_list(self, device: str) -> None:
        t = lucid.tensor([[1, 2], [3, 4]], dtype=lucid.int32, device=device)
        assert_equal_int(t, np.array([[1, 2], [3, 4]], dtype=np.int32))

    def test_dtype_override(self, device: str) -> None:
        skip_if_unsupported(device, lucid.float64)
        t = lucid.tensor([1, 2, 3], dtype=lucid.float64, device=device)
        assert t.dtype == lucid.float64

    def test_requires_grad_flag(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
        assert t.requires_grad


class TestAsTensor:
    def test_passthrough(self, device: str) -> None:
        src = lucid.tensor([1.0, 2.0], device=device)
        out = lucid.as_tensor(src)
        # Same object (no copy when no conversion needed).
        assert_close(out, src)


class TestFromNumpy:
    def test_basic(self) -> None:
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        t = lucid.from_numpy(arr)
        assert_close(t, arr)
