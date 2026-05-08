"""nn.utils — clip_grad, parametrize, weight_norm, etc."""

import numpy as np
import pytest

import lucid
import lucid.nn as nn


class TestClipGradNorm:
    def test_clamps_to_max(self) -> None:
        x = lucid.tensor([3.0, 4.0], requires_grad=True)
        (x * x).sum().backward()
        # ||grad||_2 = ||[6, 8]||_2 = 10.
        total_norm = nn.utils.clip_grad_norm_([x], max_norm=5.0)
        # ``total_norm`` may be a Tensor or a Python float depending on
        # the implementation — accept either.
        tn = total_norm.item() if hasattr(total_norm, "item") else float(total_norm)
        assert abs(tn - 10.0) < 1e-4
        # After clipping, ||grad||_2 should be ~5.
        new_norm = float(np.sqrt((x.grad.numpy() ** 2).sum()))
        assert abs(new_norm - 5.0) < 1e-4


class TestClipGradValue:
    def test_clamps_each_element(self) -> None:
        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        (x * x * x).sum().backward()
        # grad = 3x² = [3, 12].
        nn.utils.clip_grad_value_([x], clip_value=5.0)
        np.testing.assert_array_equal(x.grad.numpy(), [3.0, 5.0])


class TestCopyParametersAndBuffers:
    def test_round_trip(self) -> None:
        src = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 2))
        dst = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 2))
        for _, p in src.named_parameters():
            p._impl.copy_from(lucid.ones(p.shape, dtype=p.dtype, device=p.device)._impl)
        nn.utils.copy_parameters_and_buffers(src, dst)
        for (_, ps), (_, pd) in zip(src.named_parameters(), dst.named_parameters()):
            np.testing.assert_array_equal(ps.numpy(), pd.numpy())
        for (_, bs), (_, bd) in zip(src.named_buffers(), dst.named_buffers()):
            np.testing.assert_array_equal(bs.numpy(), bd.numpy())

    def test_missing_param_raises(self) -> None:
        src = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        dst = nn.Sequential(nn.Linear(4, 8))
        with pytest.raises(KeyError, match="present on source but missing on dest"):
            nn.utils.copy_parameters_and_buffers(src, dst)


class TestFuseConvBnEval:
    def test_matches_unfused_2d(self) -> None:
        np.random.seed(0)
        conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=True)
        bn = nn.BatchNorm2d(8)
        bn.eval(); conv.eval()
        bn.running_mean._impl.copy_from((lucid.ones(8) * 0.5)._impl)
        bn.running_var._impl.copy_from((lucid.ones(8) * 1.5)._impl)

        x = lucid.tensor(np.random.randn(2, 3, 8, 8).astype(np.float32))
        ref = bn(conv(x)).numpy()
        fused = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        out = fused(x).numpy()
        np.testing.assert_allclose(ref, out, atol=1e-5)

    def test_with_bn_affine_off(self) -> None:
        # Affine-disabled BN: scale = 1 / sqrt(σ² + ε) (no γ/β).
        np.random.seed(0)
        conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=True)
        bn = nn.BatchNorm2d(4, affine=False)
        bn.eval(); conv.eval()
        bn.running_mean._impl.copy_from((lucid.ones(4) * 0.2)._impl)

        x = lucid.tensor(np.random.randn(1, 3, 6, 6).astype(np.float32))
        ref = bn(conv(x)).numpy()
        fused = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        out = fused(x).numpy()
        np.testing.assert_allclose(ref, out, atol=1e-5)

    def test_no_bias_conv(self) -> None:
        # Conv2d(bias=False) is now accepted by the engine binding.
        np.random.seed(0)
        conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        bn = nn.BatchNorm2d(4)
        bn.eval(); conv.eval()
        bn.running_mean._impl.copy_from((lucid.ones(4) * 0.2)._impl)

        x = lucid.tensor(np.random.randn(1, 3, 6, 6).astype(np.float32))
        ref = bn(conv(x)).numpy()
        fused = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        out = fused(x).numpy()
        np.testing.assert_allclose(ref, out, atol=1e-5)
