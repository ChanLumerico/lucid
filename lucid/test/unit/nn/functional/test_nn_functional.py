"""nn.functional — numerical correctness against closed-form refs."""

import math

import numpy as np

import lucid
import lucid.nn.functional as F


class TestActivationsF:
    def test_relu(self) -> None:
        out = F.relu(lucid.tensor([-1.0, 0.0, 1.0])).numpy()
        np.testing.assert_array_equal(out, [0.0, 0.0, 1.0])

    def test_softmax_uniform(self) -> None:
        out = F.softmax(lucid.tensor([[1.0, 1.0, 1.0]]), dim=1).numpy()
        np.testing.assert_allclose(out, [[1 / 3] * 3], atol=1e-6)

    def test_log_softmax_sum_exp_one(self) -> None:
        out = F.log_softmax(lucid.tensor([[1.0, 1.0, 1.0]]), dim=1).numpy()
        # exp of log_softmax should sum to 1.
        assert abs(np.exp(out).sum() - 1.0) < 1e-6

    def test_gelu(self) -> None:
        # gelu(0) = 0.
        assert abs(F.gelu(lucid.tensor([0.0])).item()) < 1e-6

    def test_silu(self) -> None:
        # silu(x) = x * sigmoid(x); silu(0) = 0.
        assert abs(F.silu(lucid.tensor([0.0])).item()) < 1e-6

    def test_softplus(self) -> None:
        # softplus(0) = log 2.
        assert abs(F.softplus(lucid.tensor([0.0])).item() - math.log(2.0)) < 1e-5


class TestPooling:
    def test_max_pool2d_shape(self) -> None:
        out = F.max_pool2d(lucid.zeros(1, 1, 4, 4), kernel_size=2)
        assert out.shape == (1, 1, 2, 2)

    def test_avg_pool2d_value(self) -> None:
        # Average of [[1, 1], [1, 1]] over 2x2 → 1.0.
        x = lucid.ones(1, 1, 2, 2)
        out = F.avg_pool2d(x, kernel_size=2)
        assert out.item() == 1.0


class TestNormalization:
    def test_layer_norm_unit_variance(self) -> None:
        # After layer_norm the output dim has zero mean / unit variance.
        x = lucid.tensor([[1.0, 2.0, 3.0, 4.0]])
        gamma = lucid.ones(4)
        beta = lucid.zeros(4)
        out = F.layer_norm(x, [4], weight=gamma, bias=beta).numpy()
        assert abs(out.mean()) < 1e-5
        assert abs(out.std() - 1.0) < 1e-2  # bessel correction in some impls.


class TestLossesF:
    def test_mse(self) -> None:
        x = lucid.tensor([1.0, 2.0])
        y = lucid.tensor([2.0, 4.0])
        assert abs(F.mse_loss(x, y).item() - 2.5) < 1e-6

    def test_cross_entropy_uniform(self) -> None:
        x = lucid.tensor([[1.0, 1.0, 1.0]])
        y = lucid.tensor([0], dtype=lucid.int64)
        assert abs(F.cross_entropy(x, y).item() - math.log(3.0)) < 1e-5


class TestGumbelSoftmax:
    def test_soft_sums_to_one(self) -> None:
        out = F.gumbel_softmax(
            lucid.tensor([[1.0, 2.0, 3.0]]), tau=1.0, hard=False
        ).numpy()
        assert abs(out.sum() - 1.0) < 1e-5

    def test_hard_one_hot(self) -> None:
        out = F.gumbel_softmax(
            lucid.tensor([[1.0, 2.0, 3.0]]), tau=1.0, hard=True
        ).numpy()
        assert out.sum() == 1.0


class TestTripletWithDistance:
    def test_zero_when_dpos_zero(self) -> None:
        a = lucid.tensor([[1.0, 0.0]])
        p = lucid.tensor([[1.0, 0.0]])
        n = lucid.tensor([[0.0, 1.0]])
        assert (
            abs(F.triplet_margin_with_distance_loss(a, p, n, margin=1.0).item()) < 1e-6
        )


class TestFractionalPool:
    def test_frac_pool2d_output_size_shape(self) -> None:
        x = lucid.ones(2, 3, 8, 8)
        out = F.fractional_max_pool2d(x, kernel_size=2, output_size=4)
        assert tuple(out.shape) == (2, 3, 4, 4)

    def test_frac_pool2d_output_ratio_shape(self) -> None:
        x = lucid.ones(1, 2, 8, 8)
        out = F.fractional_max_pool2d(x, kernel_size=2, output_ratio=0.5)
        assert tuple(out.shape) == (1, 2, 4, 4)

    def test_frac_pool2d_deterministic_value(self) -> None:
        # sample=0 → alpha=2 → starts=[0, 2] for both H and W
        # x = arange(16) reshaped to (1,1,4,4):
        #   [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
        # pool[0,0]=max(0,1,4,5)=5  pool[0,1]=max(2,3,6,7)=7
        # pool[1,0]=max(8,9,12,13)=13  pool[1,1]=max(10,11,14,15)=15
        x = lucid.arange(16, dtype=lucid.float32).reshape(1, 1, 4, 4)
        samples = lucid.zeros(1, 1, 2)
        out = F.fractional_max_pool2d(
            x, kernel_size=2, output_size=2, _random_samples=samples
        )
        np.testing.assert_allclose(
            out.numpy().flatten(), [5.0, 7.0, 13.0, 15.0], atol=1e-6
        )

    def test_frac_pool2d_return_indices_shape(self) -> None:
        x = lucid.randn(1, 1, 6, 6)
        out, idx = F.fractional_max_pool2d(  # type: ignore[misc]
            x, kernel_size=2, output_size=3, return_indices=True
        )
        assert tuple(out.shape) == (1, 1, 3, 3)
        assert tuple(idx.shape) == (1, 1, 3, 3)

    def test_frac_pool2d_indices_in_valid_range(self) -> None:
        x = lucid.randn(1, 2, 6, 6)
        _, idx = F.fractional_max_pool2d(  # type: ignore[misc]
            x, kernel_size=2, output_size=3, return_indices=True
        )
        flat = idx.numpy().flatten()
        assert int(flat.min()) >= 0
        assert int(flat.max()) < 6 * 6

    def test_frac_pool2d_indices_point_to_max(self) -> None:
        # Verify each returned index actually points to the max of its window.
        x = lucid.randn(1, 1, 4, 4)
        samples = lucid.zeros(1, 1, 2)  # deterministic: starts=[0,2]
        out, idx = F.fractional_max_pool2d(  # type: ignore[misc]
            x,
            kernel_size=2,
            output_size=2,
            return_indices=True,
            _random_samples=samples,
        )
        x_np = x.numpy()[0, 0]  # (4, 4)
        out_np = out.numpy()[0, 0]  # (2, 2)
        idx_np = idx.numpy()[0, 0]  # (2, 2)
        for i in range(2):
            for j in range(2):
                flat = int(idx_np[i, j])
                r, c = flat // 4, flat % 4
                np.testing.assert_allclose(x_np[r, c], out_np[i, j], atol=1e-6)

    def test_frac_pool2d_backward(self) -> None:
        x = lucid.randn(1, 1, 4, 4, requires_grad=True)
        out = F.fractional_max_pool2d(x, kernel_size=2, output_size=2)
        out.sum().backward()
        assert x.grad is not None
        assert tuple(x.grad.shape) == (1, 1, 4, 4)

    def test_frac_pool2d_grad_sums_to_one_per_channel(self) -> None:
        # Each input element is the max of at most one window, so the
        # gradient of sum(output) w.r.t. x must sum to oH * oW.
        x = lucid.randn(1, 1, 4, 4, requires_grad=True)
        out = F.fractional_max_pool2d(x, kernel_size=2, output_size=2)
        out.sum().backward()
        assert x.grad is not None
        total = float(x.grad.numpy().sum())
        assert abs(total - 4.0) < 1e-5  # 2*2 output cells, each contributes 1

    def test_frac_pool2d_error_both_size_and_ratio(self) -> None:
        import pytest as _pytest

        x = lucid.ones(1, 1, 4, 4)
        with _pytest.raises(ValueError):
            F.fractional_max_pool2d(x, kernel_size=2, output_size=2, output_ratio=0.5)

    def test_frac_pool2d_error_neither_size_nor_ratio(self) -> None:
        import pytest as _pytest

        x = lucid.ones(1, 1, 4, 4)
        with _pytest.raises(ValueError):
            F.fractional_max_pool2d(x, kernel_size=2)

    def test_frac_pool3d_output_size_shape(self) -> None:
        x = lucid.ones(1, 2, 8, 8, 8)
        out = F.fractional_max_pool3d(x, kernel_size=2, output_size=4)
        assert tuple(out.shape) == (1, 2, 4, 4, 4)

    def test_frac_pool3d_output_ratio_shape(self) -> None:
        x = lucid.ones(1, 1, 8, 8, 8)
        out = F.fractional_max_pool3d(x, kernel_size=2, output_ratio=0.5)
        assert tuple(out.shape) == (1, 1, 4, 4, 4)

    def test_frac_pool3d_return_indices_shape_and_range(self) -> None:
        x = lucid.randn(1, 1, 6, 6, 6)
        out, idx = F.fractional_max_pool3d(  # type: ignore[misc]
            x, kernel_size=2, output_size=3, return_indices=True
        )
        assert tuple(out.shape) == (1, 1, 3, 3, 3)
        assert tuple(idx.shape) == (1, 1, 3, 3, 3)
        flat = idx.numpy().flatten()
        assert int(flat.min()) >= 0
        assert int(flat.max()) < 6 * 6 * 6

    def test_frac_pool3d_backward(self) -> None:
        x = lucid.randn(1, 1, 4, 4, 4, requires_grad=True)
        out = F.fractional_max_pool3d(x, kernel_size=2, output_size=2)
        out.sum().backward()
        assert x.grad is not None
        assert tuple(x.grad.shape) == (1, 1, 4, 4, 4)
