"""Tests for the lazy module variants — Conv*, ConvTranspose*, BatchNorm*, InstanceNorm*."""

from collections import OrderedDict

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.test.helpers.numerics import make_tensor


def _set_weight(target_param, src_np: np.ndarray) -> None:
    """Mirror a numpy array into a lucid Parameter without going through .data."""
    from lucid._C import engine as _C_engine
    from lucid._tensor.tensor import _impl_with_grad as _iwg

    new_impl = _C_engine.TensorImpl(src_np.copy(), _C_engine.Device.CPU, False)
    target_param._impl = _iwg(new_impl, target_param._impl.requires_grad)


# ── Lazy Conv ────────────────────────────────────────────────────────────────


class TestLazyConv:
    @pytest.mark.parametrize(
        "lazy_cls,eager_cls,input_shape,kernel,extra",
        [
            (nn.LazyConv1d, nn.Conv1d, (2, 4, 12), 3, {"padding": 1}),
            (nn.LazyConv2d, nn.Conv2d, (2, 4, 8, 8), 3, {"padding": 1}),
            (nn.LazyConv3d, nn.Conv3d, (1, 2, 4, 4, 4), 3, {"padding": 1}),
        ],
    )
    def test_lazy_conv_materializes_on_first_forward(
        self, lazy_cls, eager_cls, input_shape, kernel, extra
    ):
        m = lazy_cls(8, kernel, **extra)
        assert m.weight is None
        assert m.in_channels is None
        x = make_tensor(input_shape)
        y = m(x)
        assert m.in_channels == input_shape[1]
        assert m.weight is not None
        assert m.bias is not None
        # Output channel dim matches the requested out_channels.
        assert y.shape[1] == 8

    def test_lazy_conv2d_groups_padding_mode(self):
        m = nn.LazyConv2d(8, 3, padding="same", padding_mode="reflect", groups=2)
        x = make_tensor((1, 4, 5, 5))
        y = m(x)
        assert y.shape == (1, 8, 5, 5)
        assert m.weight.shape == (8, 2, 3, 3)
        assert "padding_mode='reflect'" in repr(m)

    def test_lazy_conv2d_state_dict_roundtrip(self):
        rng = np.random.default_rng(0)
        src = nn.Conv2d(4, 8, 3, padding=1)
        # Run a forward to ensure parameters are non-trivial values.
        src(lucid.tensor(rng.standard_normal((1, 4, 5, 5)).astype(np.float32)))
        dst = nn.LazyConv2d(8, 3, padding=1)
        result = dst.load_state_dict(src.state_dict())
        assert result.missing_keys == []
        assert result.unexpected_keys == []
        assert dst.in_channels == 4
        np.testing.assert_allclose(dst.weight.numpy(), src.weight.numpy())

    def test_lazy_conv1d_state_dict_shape_validation(self):
        # Wrong-dim weight should produce an error message, not a crash.
        dst = nn.LazyConv1d(8, 3)
        sd = OrderedDict()
        sd["weight"] = lucid.zeros(8, 4)  # 2-D, expected 3-D
        sd._metadata = {}
        with pytest.raises(RuntimeError, match="LazyConv1d"):
            dst.load_state_dict(sd, strict=False)

    def test_lazy_conv2d_out_channels_mismatch_rejected(self):
        dst = nn.LazyConv2d(8, 3)
        sd = OrderedDict()
        sd["weight"] = lucid.zeros(16, 4, 3, 3)  # out_channels mismatch
        sd._metadata = {}
        with pytest.raises(RuntimeError, match="out_channels mismatch"):
            dst.load_state_dict(sd, strict=False)


class TestLazyConvTranspose:
    @pytest.mark.parametrize(
        "lazy_cls,eager_cls,input_shape,kernel",
        [
            (nn.LazyConvTranspose1d, nn.ConvTranspose1d, (2, 4, 6), 3),
            (nn.LazyConvTranspose2d, nn.ConvTranspose2d, (2, 4, 4, 4), 3),
            (nn.LazyConvTranspose3d, nn.ConvTranspose3d, (1, 2, 3, 3, 3), 3),
        ],
    )
    def test_lazy_conv_transpose_materializes(
        self, lazy_cls, eager_cls, input_shape, kernel
    ):
        m = lazy_cls(8, kernel)
        assert m.weight is None
        x = make_tensor(input_shape)
        y = m(x)
        assert m.in_channels == input_shape[1]
        # ConvTranspose weight: (in_channels, out_channels // groups, ...)
        assert m.weight.shape[0] == input_shape[1]
        assert m.weight.shape[1] == 8

    def test_lazy_conv_transpose_rejects_string_padding(self):
        with pytest.raises(ValueError, match="string padding"):
            nn.LazyConvTranspose2d(8, 3, padding="same")

    def test_lazy_conv_transpose2d_state_dict_roundtrip(self):
        rng = np.random.default_rng(1)
        src = nn.ConvTranspose2d(4, 8, 3)
        src(lucid.tensor(rng.standard_normal((1, 4, 4, 4)).astype(np.float32)))
        dst = nn.LazyConvTranspose2d(8, 3)
        dst.load_state_dict(src.state_dict())
        assert dst.in_channels == 4
        np.testing.assert_allclose(dst.weight.numpy(), src.weight.numpy())


# ── Lazy BatchNorm / InstanceNorm ────────────────────────────────────────────


class TestLazyBatchNorm:
    @pytest.mark.parametrize(
        "lazy_cls,input_shape",
        [
            (nn.LazyBatchNorm1d, (4, 6, 5)),
            (nn.LazyBatchNorm2d, (4, 6, 4, 4)),
            (nn.LazyBatchNorm3d, (1, 6, 3, 3, 3)),
        ],
    )
    def test_lazy_bn_materializes(self, lazy_cls, input_shape):
        m = lazy_cls()
        assert m.num_features is None
        assert m._buffers["running_mean"] is None
        x = make_tensor(input_shape)
        m.train()
        m(x)
        assert m.num_features == input_shape[1]
        assert m.weight is not None
        assert m.running_mean is not None
        assert m.num_batches_tracked is not None
        assert int(m.num_batches_tracked.numpy()) == 1

    def test_lazy_bn_state_dict_roundtrip(self):
        rng = np.random.default_rng(0)
        src = nn.BatchNorm2d(6)
        src.train()
        for _ in range(3):
            src(lucid.tensor(rng.standard_normal((4, 6, 5, 5)).astype(np.float32)))
        sd = src.state_dict()
        dst = nn.LazyBatchNorm2d()
        result = dst.load_state_dict(sd)
        assert result.missing_keys == []
        assert result.unexpected_keys == []
        assert dst.num_features == 6
        np.testing.assert_allclose(dst.running_mean.numpy(), src.running_mean.numpy())
        np.testing.assert_allclose(dst.running_var.numpy(), src.running_var.numpy())

    def test_lazy_bn_affine_false(self):
        m = nn.LazyBatchNorm2d(affine=False)
        x = make_tensor((2, 4, 4, 4))
        m(x)
        assert m.weight is None
        assert m.bias is None
        # Running stats still tracked by default.
        assert m.running_mean is not None

    def test_lazy_bn_track_running_stats_false(self):
        m = nn.LazyBatchNorm2d(track_running_stats=False)
        x = make_tensor((2, 4, 4, 4))
        m(x)
        assert m._buffers["running_mean"] is None
        # state_dict reflects the absence.
        sd = m.state_dict()
        assert "running_mean" not in sd
        assert "num_batches_tracked" not in sd

    def test_lazy_bn_momentum_none_cumulative(self):
        m = nn.LazyBatchNorm2d(momentum=None)
        m.train()
        rng = np.random.default_rng(0)
        means = []
        for i in range(3):
            x = rng.standard_normal((4, 6, 4, 4)).astype(np.float32) + i
            means.append(x.mean(axis=(0, 2, 3)))
            m(lucid.tensor(x))
        expected = np.mean(means, axis=0)
        np.testing.assert_allclose(
            m.running_mean.numpy(), expected, atol=1e-4, rtol=1e-4
        )

    def test_lazy_bn_metadata_version(self):
        m = nn.LazyBatchNorm1d()
        m(make_tensor((4, 6, 5)))
        sd = m.state_dict()
        assert sd._metadata.get("", {}).get("version") == 2


class TestLazyInstanceNorm:
    @pytest.mark.parametrize(
        "lazy_cls,input_shape",
        [
            (nn.LazyInstanceNorm1d, (4, 6, 5)),
            (nn.LazyInstanceNorm2d, (4, 6, 4, 4)),
            (nn.LazyInstanceNorm3d, (1, 6, 3, 3, 3)),
        ],
    )
    def test_lazy_in_materializes(self, lazy_cls, input_shape):
        m = lazy_cls()
        m(make_tensor(input_shape))
        assert m.num_features == input_shape[1]


# ── Cross-cutting integration with state_dict v2 ─────────────────────────────


class TestLazyInModel:
    def test_lazy_chain_in_sequential(self):
        # A small CNN built entirely with lazy layers — should self-configure
        # on first forward.
        model = nn.Sequential(
            nn.LazyConv2d(8, 3, padding=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(16, 3, padding=1),
        )
        x = make_tensor((2, 3, 8, 8))
        y = model(x)
        assert y.shape == (2, 16, 8, 8)
        # All layers should now report their inferred dimensions.
        assert model[0].in_channels == 3
        assert model[1].num_features == 8
        assert model[3].in_channels == 8

    def test_lazy_model_load_from_eager_checkpoint(self):
        # Train an eager model; load its state_dict into a lazy model.
        rng = np.random.default_rng(0)
        eager = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, padding=1),
        )
        eager.train()
        for _ in range(3):
            eager(lucid.tensor(rng.standard_normal((2, 3, 5, 5)).astype(np.float32)))
        lazy = nn.Sequential(
            nn.LazyConv2d(8, 3, padding=1),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(16, 3, padding=1),
        )
        lazy.load_state_dict(eager.state_dict())
        assert lazy[0].in_channels == 3
        assert lazy[1].num_features == 8
        assert lazy[2].in_channels == 8
        # Running stats survived the round-trip.
        np.testing.assert_allclose(
            lazy[1].running_mean.numpy(), eager[1].running_mean.numpy()
        )
