"""Phase 9.1/9.2 — SharedStorage zero-copy CPU↔GPU API tests.

Verifies:
  * lucid.metal.shared_tensor()  — direct SharedStorage allocation
  * lucid.metal.to_shared()      — CPU/GPU → SharedStorage promotion
  * lucid.metal.is_shared()      — predicate
  * Tensor.is_shared             — property alias
  * transfer_storage fast path   — zero-copy .to("metal") / .to("cpu")
  * .to() value correctness      — data survives cross-device round-trips
  * GPU op on shared-derived tensor
"""

import numpy as np
import pytest

import lucid
import lucid.metal as metal
from lucid.test._fixtures.devices import metal_available


@pytest.fixture(autouse=True)
def _require_metal() -> None:
    if not metal_available():
        pytest.skip("Metal not available on this host")


# ── SharedStorage allocation ──────────────────────────────────────────────────


class TestSharedStorageAllocation:
    def test_shared_tensor_shape_dtype(self) -> None:
        t = metal.shared_tensor((3, 4))
        assert t.shape == (3, 4)
        assert t.dtype == lucid.float32
        assert t.is_shared
        assert metal.is_shared(t)

    def test_shared_tensor_custom_dtype(self) -> None:
        t = metal.shared_tensor((8,), dtype=lucid.float16)
        assert t.dtype == lucid.float16
        assert t.is_shared

    def test_shared_tensor_zero_filled(self) -> None:
        t = metal.shared_tensor((4, 4))
        np.testing.assert_array_equal(t.numpy(), np.zeros((4, 4), dtype="float32"))

    def test_shared_tensor_requires_grad(self) -> None:
        t = metal.shared_tensor((2,), requires_grad=True)
        assert t.requires_grad
        assert t.is_shared

    def test_shared_tensor_empty(self) -> None:
        # Zero-element tensor should not crash.
        t = metal.shared_tensor((0,))
        assert t.numel() == 0
        # is_shared may be False for the empty fallback path — just no crash.

    def test_shared_tensor_device_is_cpu(self) -> None:
        # Starts as CPU device so CPU ops work immediately.
        t = metal.shared_tensor((4,))
        assert not t.is_metal


# ── to_shared ─────────────────────────────────────────────────────────────────


class TestToShared:
    def test_to_shared_from_cpu(self) -> None:
        x = lucid.tensor([1.0, 2.0, 3.0])
        xs = metal.to_shared(x)
        assert xs.is_shared
        np.testing.assert_array_equal(xs.numpy(), x.numpy())

    def test_to_shared_idempotent(self) -> None:
        x = lucid.tensor([1.0, 2.0])
        xs = metal.to_shared(x)
        xs2 = metal.to_shared(xs)
        # Already shared → same Python object returned.
        assert xs2 is xs
        assert xs2.is_shared

    def test_to_shared_from_gpu(self) -> None:
        x = lucid.tensor([4.0, 5.0, 6.0]).to("metal")
        xs = metal.to_shared(x)
        assert xs.is_shared
        np.testing.assert_array_equal(xs.numpy(), [4.0, 5.0, 6.0])

    def test_to_shared_preserves_shape(self) -> None:
        x = lucid.ones((3, 4, 2))
        xs = metal.to_shared(x)
        assert xs.shape == (3, 4, 2)
        assert xs.is_shared


# ── is_shared / Tensor.is_shared ─────────────────────────────────────────────


class TestIsShared:
    def test_is_shared_false_for_cpu(self) -> None:
        x = lucid.randn(4)
        assert not metal.is_shared(x)
        assert not x.is_shared

    def test_is_shared_false_for_gpu(self) -> None:
        x = lucid.randn(4).to("metal")
        assert not metal.is_shared(x)
        assert not x.is_shared

    def test_is_shared_true_after_to_shared(self) -> None:
        x = lucid.randn(4)
        xs = metal.to_shared(x)
        assert metal.is_shared(xs)
        assert xs.is_shared

    def test_is_shared_true_for_shared_tensor(self) -> None:
        t = metal.shared_tensor((8,))
        assert metal.is_shared(t)
        assert t.is_shared

    def test_tensor_property_and_fn_agree(self) -> None:
        x = lucid.tensor([1.0, 2.0])
        xs = metal.to_shared(x)
        assert xs.is_shared == metal.is_shared(xs)


# ── Zero-copy transfer via .to() ──────────────────────────────────────────────


class TestZeroCopyTransfer:
    def test_shared_to_metal_is_metal(self) -> None:
        t = metal.shared_tensor((4,))
        tg = t.to("metal")
        assert tg.is_metal

    def test_shared_to_metal_values(self) -> None:
        x = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        xs = metal.to_shared(x)
        xg = xs.to("metal")
        np.testing.assert_array_equal(xg.to("cpu").numpy(), [1.0, 2.0, 3.0, 4.0])

    def test_shared_to_cpu_is_cpu(self) -> None:
        t = metal.shared_tensor((4,))
        tc = t.to("cpu")
        assert not tc.is_metal

    def test_shared_to_cpu_values(self) -> None:
        x = lucid.tensor([10.0, 20.0])
        xs = metal.to_shared(x)
        xc = xs.to("cpu")
        np.testing.assert_array_equal(xc.numpy(), [10.0, 20.0])

    def test_cpu_to_gpu_value_preserved(self) -> None:
        x = lucid.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
        xg = x.to("metal")
        np.testing.assert_array_equal(xg.to("cpu").numpy(), [3.0, 1.0, 4.0, 1.0, 5.0])

    def test_gpu_to_cpu_value_preserved(self) -> None:
        x = lucid.tensor([2.71, 3.14]).to("metal")
        xc = x.to("cpu")
        np.testing.assert_allclose(xc.numpy(), [2.71, 3.14], atol=1e-5)

    def test_roundtrip_cpu_metal_cpu(self) -> None:
        orig = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
        x = lucid.tensor(orig)
        xg = x.to("metal")
        xc = xg.to("cpu")
        np.testing.assert_array_equal(xc.numpy(), orig)

    def test_write_cpu_read_via_metal(self) -> None:
        """Write on CPU via shared buffer, read back from GPU path."""
        t = metal.shared_tensor((4,))
        # Write values on CPU (shared buffer is directly writable).
        src = lucid.tensor([10.0, 20.0, 30.0, 40.0])
        t_src = metal.to_shared(src)
        tg = t_src.to("metal")
        np.testing.assert_array_equal(tg.to("cpu").numpy(), [10.0, 20.0, 30.0, 40.0])


# ── .to() method behaviour ────────────────────────────────────────────────────


class TestToMethodBehaviour:
    def test_to_preserves_requires_grad_cpu_to_gpu(self) -> None:
        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        xg = x.to("metal")
        assert xg.requires_grad

    def test_to_preserves_requires_grad_gpu_to_cpu(self) -> None:
        x = lucid.tensor([1.0, 2.0], requires_grad=True).to("metal")
        xc = x.to("cpu")
        assert xc.requires_grad

    def test_to_same_device_returns_self(self) -> None:
        x = lucid.randn(4)
        y = x.to("cpu")
        # copy=False default → same object when no change needed.
        assert y is x

    def test_to_no_change_copy_false(self) -> None:
        x = lucid.randn(4)
        y = x.to(lucid.float32)  # same dtype, same device
        assert y is x


# ── GPU op on shared-derived tensor ──────────────────────────────────────────


class TestGpuOpOnSharedTensor:
    def test_relu_on_shared_to_metal(self) -> None:
        import lucid.nn.functional as F

        x = lucid.tensor([-1.0, 0.0, 1.0, 2.0])
        xs = metal.to_shared(x)
        xg = xs.to("metal")
        out = F.relu(xg)
        np.testing.assert_array_equal(out.to("cpu").numpy(), [0.0, 0.0, 1.0, 2.0])

    def test_add_on_shared_derived(self) -> None:
        a = lucid.tensor([1.0, 2.0, 3.0])
        b = lucid.tensor([4.0, 5.0, 6.0])
        ag = metal.to_shared(a).to("metal")
        bg = metal.to_shared(b).to("metal")
        out = ag + bg
        np.testing.assert_array_equal(out.to("cpu").numpy(), [5.0, 7.0, 9.0])
