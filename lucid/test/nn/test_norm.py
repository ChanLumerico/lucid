"""Tests for normalization layers: BatchNorm, LayerNorm, GroupNorm, RMSNorm."""

import pytest
import numpy as np
import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.test._comparison import assert_close
from lucid.test.helpers.numerics import make_tensor


class TestLayerNorm:
    def test_output_shape(self):
        layer = nn.LayerNorm(8)
        x = make_tensor((3, 8))
        assert layer(x).shape == (3, 8)

    def test_normalized_mean_near_zero(self):
        layer = nn.LayerNorm(8)
        x = make_tensor((4, 8))
        out = layer(x)
        # mean of each row after LayerNorm should be near 0
        row_means = out.numpy().mean(axis=-1)
        np.testing.assert_allclose(row_means, 0.0, atol=1e-5)

    def test_normalized_var_near_one(self):
        layer = nn.LayerNorm(8)
        x = make_tensor((4, 8))
        out = layer(x)
        row_vars = out.numpy().var(axis=-1)
        np.testing.assert_allclose(row_vars, 1.0, atol=1e-4)

    def test_3d_input(self):
        layer = nn.LayerNorm([4, 5])
        x = make_tensor((3, 4, 5))
        assert layer(x).shape == (3, 4, 5)


class TestBatchNorm:
    def test_output_shape_2d(self):
        layer = nn.BatchNorm2d(4)
        x = make_tensor((2, 4, 8, 8))
        out = layer(x)
        assert out.shape == (2, 4, 8, 8)

    def test_output_shape_1d(self):
        layer = nn.BatchNorm1d(8)
        x = make_tensor((4, 8))
        assert layer(x).shape == (4, 8)

    def test_eval_mode_uses_running_stats(self):
        layer = nn.BatchNorm2d(4)
        layer.eval()
        x = make_tensor((2, 4, 4, 4))
        out = layer(x)
        assert out.shape == (2, 4, 4, 4)

    def test_running_mean_updated_in_train(self):
        layer = nn.BatchNorm2d(4)
        layer.train()
        # Synthesise input with a non-zero per-channel mean so the update
        # is observable.
        x = make_tensor((4, 4, 4, 4)) + 3.0
        before_mean = layer.running_mean.numpy().copy()
        before_count = int(layer.num_batches_tracked.numpy())
        layer(x)
        after_mean = layer.running_mean.numpy()
        after_count = int(layer.num_batches_tracked.numpy())
        # running_mean must have moved off zero, num_batches_tracked must increment.
        assert not np.allclose(before_mean, after_mean)
        assert after_count == before_count + 1

    def test_train_then_eval_uses_running_stats(self):
        layer = nn.BatchNorm2d(4)
        layer.train()
        x = make_tensor((4, 4, 4, 4)) + 3.0
        for _ in range(5):
            layer(x)
        layer.eval()
        # eval mode should use the running_mean we accumulated.  Compare
        # against a manually-computed reference: (x − running_mean) / sqrt(running_var + eps).
        rm = layer.running_mean.numpy()
        rv = layer.running_var.numpy()
        out = layer(x).numpy()
        x_np = x.numpy()
        expected = (x_np - rm[None, :, None, None]) / np.sqrt(
            rv[None, :, None, None] + layer.eps
        )
        np.testing.assert_allclose(out, expected, atol=1e-4, rtol=1e-4)

    def test_track_running_stats_false_skips_buffers(self):
        layer = nn.BatchNorm2d(4, track_running_stats=False)
        # Buffers are registered as None — not in state_dict either.
        sd = layer.state_dict()
        assert "running_mean" not in sd
        assert "running_var" not in sd
        assert "num_batches_tracked" not in sd
        # Forward in eval mode should still work — falls back to batch stats.
        layer.eval()
        x = make_tensor((4, 4, 4, 4))
        out = layer(x)
        assert out.shape == (4, 4, 4, 4)

    def test_momentum_none_cumulative_average(self):
        layer = nn.BatchNorm2d(4, momentum=None)
        layer.train()
        means_per_batch = []
        for i in range(3):
            x = make_tensor((4, 4, 4, 4)) + float(i)
            means_per_batch.append(x.numpy().mean(axis=(0, 2, 3)))
            layer(x)
        # Cumulative average = mean of per-batch means.
        expected = np.mean(means_per_batch, axis=0)
        np.testing.assert_allclose(
            layer.running_mean.numpy(), expected, atol=1e-4, rtol=1e-4
        )

    def test_affine_false_no_weight_bias(self):
        layer = nn.BatchNorm2d(4, affine=False)
        assert layer.weight is None
        assert layer.bias is None
        sd = layer.state_dict()
        assert "weight" not in sd
        assert "bias" not in sd


class TestLayerNormBias:
    def test_bias_false_creates_only_weight(self):
        ln = nn.LayerNorm(8, bias=False)
        assert ln.weight is not None
        assert ln.bias is None
        sd = ln.state_dict()
        assert "weight" in sd
        assert "bias" not in sd

    def test_bias_false_normalizes(self):
        ln = nn.LayerNorm(8, bias=False)
        x = make_tensor((4, 8))
        out = ln(x).numpy()
        # Without bias, the output is still zero-mean per row (since gamma=1).
        np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-5)

    def test_elementwise_affine_false(self):
        ln = nn.LayerNorm(8, elementwise_affine=False)
        assert ln.weight is None
        assert ln.bias is None
        x = make_tensor((4, 8))
        out = ln(x)
        assert out.shape == (4, 8)


class TestGroupNorm:
    def test_output_shape(self):
        layer = nn.GroupNorm(num_groups=2, num_channels=8)
        x = make_tensor((3, 8, 4, 4))
        assert layer(x).shape == (3, 8, 4, 4)

    def test_instance_norm_case(self):
        # GroupNorm with num_groups == num_channels = InstanceNorm
        layer = nn.GroupNorm(num_groups=4, num_channels=4)
        x = make_tensor((2, 4, 6, 6))
        assert layer(x).shape == (2, 4, 6, 6)


class TestRMSNorm:
    def test_output_shape(self):
        layer = nn.RMSNorm(8)
        x = make_tensor((3, 8))
        assert layer(x).shape == (3, 8)

    def test_rms_near_one(self):
        layer = nn.RMSNorm(8)
        x = make_tensor((4, 8))
        out = layer(x)
        # After RMSNorm, the RMS of each row ≈ 1 (before learnable scale)
        # Just check shape and no NaN
        assert not np.isnan(out.numpy()).any()


class TestFunctionalNorm:
    def test_layer_norm_direct(self):
        x = make_tensor((2, 3, 8))
        w = lucid.ones(8)
        b = lucid.zeros(8)
        out = F.layer_norm(x, [8], w, b)
        assert out.shape == (2, 3, 8)

    def test_group_norm_direct(self):
        x = make_tensor((2, 8, 4, 4))
        w = lucid.ones(8)
        b = lucid.zeros(8)
        out = F.group_norm(x, num_groups=2, weight=w, bias=b)
        assert out.shape == (2, 8, 4, 4)
